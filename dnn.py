import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, n_feature, n_output, hidden_layers=[20, 20]):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feature, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DNN(object):
    def __init__(self, epochs=10, learning_rate=0.0005, hidden_layers=[20, 20], verbose=0):
        self.epochs = epochs
        self.lr = learning_rate
        self.hidden_layers = hidden_layers
        self.verbose = verbose
        self.net = None

    def fit(self, X, y, batch_size=4):
        n_feature = len(X.columns)
        n_output = len(np.unique(y))
        self.net = Net(n_feature, n_output, self.hidden_layers)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        trainset = TensorDataset(torch.from_numpy(X_train.values.astype(float)).float(),
                                 torch.from_numpy(y_train.ravel()))
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        validset = TensorDataset(torch.from_numpy(X_valid.values.astype(float)).float(),
                                 torch.from_numpy(y_valid.ravel()))
        validloader = DataLoader(validset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)  # , momentum=0.9

        min_valid_loss = float('inf')
        count = 0
        for epoch in range(self.epochs):
            train_loss = []
            valid_loss = []
            for i, (inputs, labels) in enumerate(trainloader):
                optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

            for i, (inputs, labels) in enumerate(validloader):
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                valid_loss.append(loss.item())

            if self.verbose:
                print('[%d] train_loss: %.3f valid_loss: %.3f' %
                      (epoch + 1, np.mean(train_loss), np.mean(valid_loss)))
            if np.mean(valid_loss) < min_valid_loss:
                torch.save(self.net.state_dict(), 'checkpoint.pt')
                min_valid_loss = np.mean(valid_loss)
                count = 0
            else:
                count += 1

            if count > 2:
                break

        self.net.load_state_dict(torch.load('checkpoint.pt'))

    def predict_proba(self, X):
        return F.softmax(self.net.forward(torch.from_numpy(X.values.astype(float)).float()), dim=1).detach().numpy()

    def predict(self, X):
        return self.net.forward(torch.from_numpy(X.values.astype(float)).float()).argmax(dim=1).detach().numpy()
