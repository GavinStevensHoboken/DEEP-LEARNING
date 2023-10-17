! pip install torchinfo
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Build a simple conv net:
class SimpleConvnet(nn.Module):
    def __init__(self):
        super(SimpleConvnet, self).__init__()
        # conv net
        self.convnet = nn.Sequential(
            # input (num_batch, 1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),  # (num_batch, 32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  #(num_batch, 32, 13, 13)  if stride is not provided, it will default to kernel_size
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # (num_batch, 64, 11, 11)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (num_batch, 64, 5, 5)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),  # (num_batch, 64, 3, 3)
            nn.Flatten()
        )
        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features=576, out_features=64),  # 576 = 64 * 3 * 3
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10)
        )
    def forward(self, x):
        x = self.convnet(x)
        x = self.classifier(x)

        return x

visual_model = SimpleConvnet()
summary(visual_model, (10, 1, 28, 28))

#Load MINST Dataset from google drive
# load from torch vision package
train_dataset = torchvision.datasets.MNIST('./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST('./data',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True)
valid_loader = DataLoader(dataset=test_dataset,
                          batch_size=32,
                          shuffle=False)

#Train function
def train(batch_size, num_epochs, learning_rate, model, train_loader, valid_loader, device):
  # move the model to device
  model = model.to(device)  # move the model to gpu or cpu

  # history
  history = {'train_loss': [],
             'train_acc': [],
             'valid_loss': [],
             'valid_acc': []}

  # set up loss function and optimizer
  criterion = nn.CrossEntropyLoss()  # the CrossEntropyLoss() will provide the softmax for us
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # pass in the parameters to be updated and learning rate

  # traning loop
  print('Training Starts:')
  num_total_steps = len(train_loader)
  for epoch in range(num_epochs):
      model.train()  # start to train the model, activate training behavior
      train_loss = 0
      train_acc = 0
      for i, (images, labels) in enumerate(train_loader):
          # reshape images
          images = images.to(device)  # move batch to device
          labels = labels.to(device)  # move to device
          # forward
          outputs = model(images)  # forward
          cur_train_loss = criterion(outputs, labels)  # loss
          # backward
          cur_train_loss.backward()  # run back propagation
          optimizer.step()  # optimizer update all model parameters
          optimizer.zero_grad()  # set gradient to zero, avoid gradient accumulating
          # loss
          train_loss += cur_train_loss.item()
          # acc
          _, pred_class = torch.max(outputs, 1)
          train_acc += (pred_class == labels).float().mean().item()

      # valid
      model.eval()  # start to train the model, activate training behavior
      with torch.no_grad():  # tell pytorch not to update parameters
          val_loss = 0
          val_acc = 0
          for images, labels in valid_loader:
              # calculate validation loss
              images = images.to(device)
              labels = labels.to(device)
              outputs = model(images)
              cur_valid_loss = criterion(outputs, labels)
              val_loss += cur_valid_loss.item()
              _, pred_class = torch.max(outputs, 1)
              val_acc += (pred_class == labels).float().mean().item()
      
      # print & record
      train_loss = train_loss / len(train_loader)
      train_acc = train_acc / len(train_loader)
      val_loss = val_loss / len(valid_loader)
      val_acc = val_acc / len(valid_loader)
      print(f"Epoch:{epoch + 1} / {num_epochs}, train loss:{train_loss:.5f}, train acc: {train_acc:.5f}, valid loss:{val_loss:.5f}, valid acc:{val_acc:.5f}")
      history['train_loss'].append(train_loss)
      history['train_acc'].append(train_acc)
      history['valid_loss'].append(val_loss)
      history['valid_acc'].append(val_acc)

  return history

#Train the model
model = SimpleConvnet()

history = train(batch_size=32,
                num_epochs=10,
                learning_rate=0.005,
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                device=device)

#Plot
plt.plot(history['train_acc'], label='Train')
plt.plot(history['valid_acc'], label='Valid')
plt.legend()
plt.plot

plt.plot(history['train_loss'], label='Train')
plt.plot(history['valid_loss'], label='Valid')
plt.legend()
plt.plot
