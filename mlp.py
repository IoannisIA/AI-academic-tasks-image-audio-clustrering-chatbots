import pandas as pd
import torch as torch
from sklearn.model_selection import train_test_split
from torch import nn, optim


input_size = 784
hidden_sizes = [256, 256, 6, 5, 5, 5]
output_size = 10
batch_size = 100
epochs = 10
lr = 0.1
momentum = 0.9

data = pd.read_csv("mnist.csv")

X = data.loc[:,data.columns != "label"].values/255   #Normalizing the values
Y = data.label.values
features_train, features_test, targets_train, targets_test = train_test_split(X,Y,test_size=0.2, random_state=42)

X_train = torch.from_numpy(features_train)
X_test = torch.from_numpy(features_test)

Y_train = torch.from_numpy(targets_train).type(torch.LongTensor)
Y_test = torch.from_numpy(targets_test).type(torch.LongTensor)

train = torch.utils.data.TensorDataset(X_train,Y_train)
test = torch.utils.data.TensorDataset(X_test,Y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)



class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        #nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.xavier_normal_(self.fc1.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_sizes[0], output_size)
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.fc1(x)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.softmax(output)
        return output


model = Feedforward(input_size, hidden_sizes, output_size)

model.double()

criterion = nn.CrossEntropyLoss()

images, labels = next(iter(train_loader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

model.train()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_loader)))

model.eval()

correct_count, all_count = 0, 0
for images, labels in test_loader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count / all_count))