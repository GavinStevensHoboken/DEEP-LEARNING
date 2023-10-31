from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My Drive/BIA667/

import pandas as pd
import numpy as np

data = pd.read_csv("Processed_S&P.csv", parse_dates=[0])
data = data[["Date", "Close", "Volume", \
             "mom", "mom1", "mom2", "mom3",\
            "ROC_5","ROC_10","ROC_15","ROC_20",\
            "EMA_10","EMA_20","EMA_50"]]

# drop NA
data = data.dropna(axis = 0)
print("Start date: ", data["Date"].min())
print("End date: ", data["Date"].max())

data = data.sort_values(by = "Date")

data.head()

from matplotlib import pyplot as plt
data.loc[0:500]["Close"].plot(kind="line", figsize=(10,5), title = "Indices - Close")
plt.legend()
plt.show()

data.loc[0:500]["Volume"].plot(kind="line", figsize=(10,5), title = "Volume Change")
plt.legend()
plt.show()

#preparing data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
pre_data_train = data[data["Date"]<="2016-05-06"]
pre_data_test = data[data["Date"]>"2016-05-06"]
data_train=pre_data_train[["Close", "Volume", \
             "mom", "mom1", "mom2", "mom3",\
            "ROC_5","ROC_10","ROC_15","ROC_20",\
            "EMA_10","EMA_20","EMA_50"]]
data_test=pre_data_test[["Close", "Volume", \
             "mom", "mom1", "mom2", "mom3",\
            "ROC_5","ROC_10","ROC_15","ROC_20",\
            "EMA_10","EMA_20","EMA_50"]]
scaler = StandardScaler().fit(data_train)
x_transformed=scaler.transform(data[["Close", "Volume", \
             "mom", "mom1", "mom2", "mom3",\
            "ROC_5","ROC_10","ROC_15","ROC_20",\
            "EMA_10","EMA_20","EMA_50"]])
scaler.transform(data_test)
data_train["Close"].values[0]
# Define a function to generate samples 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import itertools

def transform_data(data, feature_cols, target_col, cut_off_index, lookback=5):
    
    X, Y, Y_binary, x_scaler, y_scaler = None, None, None, None, None
    
    # add your code here
    train_sub = data[data["Date"]<="2016-05-06"]
    x_train_sub = train_sub[feature_cols]
    y_train_sub = train_sub[[target_col]]
    x_scaler = StandardScaler().fit(x_train_sub)
    y_scaler = StandardScaler().fit(y_train_sub)
    X_transformed = x_scaler.transform(data[feature_cols])
    Y_transformed = y_scaler.transform(data[[target_col]])
    X = np.array([X_transformed[i:i+lookback].tolist() for i in range(len(data)-5)])
    Y = Y_transformed[lookback:]
    Y_binary=list(map(lambda next:1 if(next[0]>next[1]) else 0, zip(data[target_col][1:],data[target_col][:-1])))[4:]
    return X, Y, Y_binary, x_scaler, y_scaler


# Transform data
np.set_printoptions(precision = 3)

feature_cols = ["Close","Volume", "mom", "mom1", "mom2", "mom3",\
            "ROC_5","ROC_10","ROC_15","ROC_20",\
            "EMA_10","EMA_20","EMA_50"]  # feel free to add your own features, e.g. volume, day, week, etc.
target_col = "Close"
lookback = 5
train_test_cut_off = int(len(data)*0.8)


X, Y, Y_binary,x_scaler, y_scaler = transform_data(data, feature_cols, target_col, \
                                                   train_test_cut_off, lookback=lookback)
print("Total samples: {0}, train: {1}, test: {2}\n".\
      format(len(X), train_test_cut_off, len(X)- train_test_cut_off))
print("Show a few observations:")
print("Before transformation:")
print(data.iloc[0:7,0:3])
print("\nAfter transformation:")
print("X:")
print(X[0:2,:,0:2])
print("\nY - Regression:")
print(Y[0:2])
print("\nY - Classification: buy or hold")
print(Y_binary[0:2])

# Define a function to calculate a baseline on the testing data

def evaluate_naive_method(test_X, test_Y, test_Y_binary):
    
    # add your code here
    mae = 0
    predict = []
    mae_binary = []
    same = 0
    for i in range(len(test_Y)):
      mae += abs(test_Y[i]-test_X[i]).mean(axis=0)
      predict = predict+[mae]
          
    mae = mae/len(test_Y)
    for i in range(len(test_X)):
      if predict[i]> test_X[i][-1]:
        mae_binary.append(1)
      else:
        mae_binary.append(0)

    for i in range(len(mae_binary)):
      if mae_binary[i] == test_Y_binary[i]:
        same += 1
    acc = same / len(test_Y_binary)

    
    return mae, acc

test_X = X[train_test_cut_off:]
test_Y = Y[train_test_cut_off:]
test_Y_binary = Y_binary[train_test_cut_off:]
mae, acc = evaluate_naive_method(test_X[:,:,0], test_Y,test_Y_binary)
print("mae: {0:.3f}, acc: {1:.3f}".format(mae, acc))

test_X = X[train_test_cut_off:]
test_Y = Y[train_test_cut_off:]
test_Y_binary = Y_binary[train_test_cut_off:]
mae, acc = evaluate_naive_method(test_X[:,:,0], test_Y,test_Y_binary)
print("mae: {0:.3f}, acc: {1:.3f}".format(mae, acc))

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class SP_Model(nn.Module):
    
    # add your model here
    def __init__(self, input_size, embedding_dim, output_size, hidden_dim, n_layers):
        super(SP_Model, self).__init__()

        # Defining some parameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # Embedding Layer
        # self.emb = nn.Embedding(input_size, embedding_dim, padding_idx = 0)

        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # linear  layer
        self.fc = nn.Linear(128, output_size)

    
    def forward(self, x):
        
        batch_size = x.size(0)
        print("batch_size",batch_size)
        # embedding = self.emb(x)
        
        # Passing in the input and hidden state into the model and obtaining outputs
        rnn_out, hidden_out = self.rnn(x)  
        # run_out shape: [bacth_size, sequence length, hidden_dim]

        # Select which time step data you want for linear layers
        out = self.fc(rnn_out[:,-1])

        #or you want to use hidden_ as an input for the next layer
        #out = self.fc(hidden_out)
        
        return out


rnn = nn.RNN(65, 65, 2)
input = torch.randn(5,65)
runout, hiddenout = rnn(rr1)


from torchinfo import summary 
batch_size = 128
cols = 65
model = SP_Model(input_size = 65, embedding_dim=66, output_size=batch_size, hidden_dim=1, n_layers=1)
# summary(model,input_size=(batch_size,605),dtypes=[torch.long])
class SP_dataset(Dataset):
    
    # define your dataset
    def __init__(self, featuers, labels):
        self.length = len(labels)
        self.features = torch.FloatTensor(featuers)
        self.labels = torch.Tensor(labels)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return self.length


# datasets
train_test_cut_off = 1536
x_train = X[:train_test_cut_off]
x_train=[np.concatenate((x_train[i][0],x_train[i][1],x_train[i][2],x_train[i][3],x_train[i][4])).tolist() for i in range(len(x_train))]
y_train = Y[:train_test_cut_off]
y_train = y_train.reshape(y_train.shape[0])
x_test = X[train_test_cut_off:1920]
x_test = [np.concatenate((x_test[i][0],x_test[i][1],x_test[i][2],x_test[i][3],x_test[i][4])).tolist() for i in range(len(x_test))]
y_test = Y[train_test_cut_off:1920]
y_test = y_test.reshape(y_test.shape[0])
train_dataset = SP_dataset(x_train, y_train)
test_dataset = SP_dataset(x_test, y_test)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for i,j in train_loader:
  print(i)

# Define a function to train the model 
def train_model(model, train_dataset, test_dataset, device, binary_pred = False,\
                lr=0.0005, epochs=20, batch_size=32):
    
    # define training function
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
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

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
            print('x',x)
            print('---------')
            x = x.to(device)
            y = y.to(device)
            # forward
            outputs = model(x).view(-1)
            pred = torch.round(torch.sigmoid(outputs))
            cur_train_loss = criterion(outputs, y)
            cur_train_acc = (pred == y).sum().item() / batch_size
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
                cur_test_acc = (pred == y).sum().item() / batch_size 
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
        print(f"Epoch:{epoch + 1} / {epochs}, train loss:{train_loss:.5f} train_acc:{train_acc:.5f}, valid loss:{val_loss:.5f} valid acc:{val_acc:.5f}")
    
    return history

history = train_model(model=model,
                      train_dataset=train_dataset,
                      test_dataset=test_dataset,
                      device=device,
                      lr=0.0005,
                      epochs=30,
                      batch_size=128)

# A sample of binary model training curves

plot_history(hista, binary_predict = True)
# A sample of regression model training curve

plot_history(histb, binary_predict = False)
