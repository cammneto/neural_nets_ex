import torch
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

breast = datasets.load_breast_cancer()

inputs = breast.data
outputs = breast.target

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.25)

x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)

dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

network = nn.Sequential(nn.Linear(in_features=30, out_features=16), nn.Sigmoid(), nn.Linear(16, 16), nn.Sigmoid(), nn.Linear(16,1), nn.Sigmoid())

loss_function = nn.BCELoss()
optmizer = torch.optim.Adam(network.parameters(), lr = 0.001)

epochs = 1000
for epoch in range(epochs):
    running_loss = 0.
    for data in train_loader:
        inputs, outputs = data
        outputs=outputs.unsqueeze(1)
        optmizer.zero_grad()
        predictions = network.forward(inputs)
        loss = loss_function(predictions, outputs)
        loss.backward()
        optmizer.step()
        running_loss += loss.item()
    print('Epoch:' + str(epoch+1) + ' -- loss:' + str(running_loss/len(train_loader)))

print(network.eval())
x_test = torch.tensor(x_test, dtype=torch.float)
predictions = network.forward(x_test)
predictions = np.array(predictions > 0.5)

print('accuracy', accuracy_score(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

mapplot=sns.heatmap(cm, annot=True)
plt.show()

