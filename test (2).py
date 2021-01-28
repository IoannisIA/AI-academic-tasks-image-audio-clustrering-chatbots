import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim

train_batch_size =64
test_batch_size =64


# ################# ## ################# # Loading the data # ################# ## ################# #

train = pd.read_csv("train.csv")


X = train.loc[:,train.columns != "label"].values/255   #Normalizing the values
Y = train.label.values

features_train, features_test, targets_train, targets_test = train_test_split(X, Y, test_size=0.2, random_state=42)


X_train = torch.from_numpy(features_train)
X_test = torch.from_numpy(features_test)

Y_train = torch.from_numpy(targets_train).type(torch.LongTensor)
Y_test = torch.from_numpy(targets_test).type(torch.LongTensor)

train = torch.utils.data.TensorDataset(X_train, Y_train)
test = torch.utils.data.TensorDataset(X_test, Y_test)


train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=False)


# ################# ## ################# # deedewwewef # ################# ## ################# #


# Hyperparameters for our network
input_size = 784
hidden_size = 10
output_size = 10
# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_size),
                      nn.ReLU(),
                      nn.Linear(hidden_size, output_size),
                      nn.Softmax(dim=1))

# Define the loss
criterion = nn.NLLLoss()
# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.1)
epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(-1,1,1,784)


        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(train_loader)}")