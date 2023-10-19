import os
import numpy as np
import torch
import torch.nn as nn
import keras  # only for loading the dataset and preprocessing
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from gensim.models import word2vec

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10_000) # only keep the top 10_000 most frequently occuring words

# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print(decoded_review)

from matplotlib import pyplot as plt
plt.hist ([len(doc) for doc in train_data], bins = 100)
plt.show()

DOC_LEN = 500

train_x = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=DOC_LEN)

test_x = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=DOC_LEN)

print(train_x.shape)
print(test_x.shape)

class IMDB_dataset(Dataset):
    def __init__(self, featuers, labels):
        self.length = len(labels)
        self.features = torch.IntTensor(featuers)
        self.labels = torch.Tensor(labels)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return self.length        

# datasets
train_dataset = IMDB_dataset(train_x, train_labels)
test_dataset = IMDB_dataset(test_x, test_labels)

train_dataset.features.size()


class TextCNN(nn.Module):
    def __init__(self, num_words_in_dict, embedding_dim, dropout_ratio):
        super(TextCNN, self).__init__()
        self.num_words_in_dict = num_words_in_dict
        self.embedding_dim = embedding_dim
        self.dropout_ratio = dropout_ratio
        
        # embedding
        self.embedding = nn.Embedding(num_embeddings=num_words_in_dict, embedding_dim=embedding_dim)  # (-1, DOC_LEN, embedding_dim), num_embedding: embedding dict size, embedding_dim: length of embedding vector
        
        # 1D CNN
        # unigram
        self.unigram = nn.Sequential(# input (-1, embedding_dim, DOC_LEN)
        nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=1),  # (-1, 64, DOC_LEN)
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=DOC_LEN),  # (-1, 64, 1)
        nn.Flatten()  # (-1, 64 * 1)
        )
        # bigram
        self.bigram = nn.Sequential(# input (-1, embedding_dim, DOC_LEN)
        nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=2),  # (-1, 64, DOC_LEN-2+1) ??? why DOC_LEN - 2 + 1
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=DOC_LEN - 2 + 1),  # (-1, 64, 1)
        nn.Flatten()  # (-1, 64 * 1)
        )
        # trigram
        self.trigram = nn.Sequential(# input (-1, embedding_dim, DOC_LEN)
        nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3),  # (-1, 64, DOC_LEN-3+1)
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=DOC_LEN - 3 + 1),  # (-1, 16, 7)
        nn.Flatten()  # (-1, 64 * 1)
        )
        # simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(in_features=64*3, out_features=1)
        )
    def forward(self, x):
        # get embedding
        x = self.embedding(x)
        # make sure we are convolving on each word
        x = torch.transpose(x, dim0=1, dim1=2)  # (-1, DOC_LEN, embedding_dim): embedding on 1(DOC_LEN) & 2(embedding_dim) dims
        # 1d cnn output
        uni_gram_output = self.unigram(x)
        bi_gram_output = self.bigram(x)
        tri_gram_output = self.trigram(x)
        # concatenate
        x = torch.cat((uni_gram_output, bi_gram_output, tri_gram_output), dim=1)
        # classifier
        x = self.classifier(x)

        return x



model=TextCNN(10000,100,0.5)


def train_model(model, train_dataset, test_dataset, device, lr=0.0001, epochs=20, batch_size=32):
    # construct dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # move model to device
    model = model.to(device)

    # history
    history = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': []}

    # setup loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    print('Training Start')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0
        for x, y in train_loader:
            # move data to device
            x = x.to(device)
            y = y.to(device)
            # forward
            outputs = model(x).view(-1)
            pred = torch.round(torch.sigmoid(outputs))
            cur_train_loss = criterion(outputs, y)
            cur_train_acc = (pred == y).float().mean().item() 
            # backward
            cur_train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # loss and acc
            train_loss += cur_train_loss
            train_acc += cur_train_acc

        # test start
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                # move
                x = x.to(device)
                y = y.to(device)
                # predict
                outputs = model(x).view(-1)
                pred = torch.round(torch.sigmoid(outputs))
                cur_test_loss = criterion(outputs, y)
                cur_test_acc = (pred == y).float().mean().item() 
                # loss and acc
                test_loss += cur_test_loss
                test_acc += cur_test_acc

        # epoch output
        train_loss = (train_loss/len(train_loader)).item()
        train_acc = train_acc/len(train_loader)
        val_loss = (test_loss/len(test_loader)).item()
        val_acc = test_acc/len(test_loader)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(val_loss)
        history['test_acc'].append(val_acc)
        print(f"Epoch:{epoch + 1} / {epochs}, train loss:{train_loss:.4f} train_acc:{train_acc:.4f}, valid loss:{val_loss:.4f} valid acc:{val_acc:.5f}")
    
    return history

history = train_model(model=model,
                      train_dataset=train_dataset,
                      test_dataset=test_dataset,
                      device=device,
                      lr=0.0005,
                      epochs=30,
                      batch_size=128)

plt.plot(range(1, 31), history['train_loss'], label='Train')
plt.plot(range(1, 31), history['test_loss'], label='Valid')
plt.legend()
plt.plot()

plt.plot(range(1, 31), history['train_acc'], label='Train')
plt.plot(range(1, 31), history['test_acc'], label='Valid')
plt.legend()
plt.plot()
