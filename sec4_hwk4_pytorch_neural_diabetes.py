import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df1 = pd.read_csv('diabetes.csv')

inputs = df1.iloc[:,:-1].values
#print(inputs.shape)
#print('-----')
scaler = MinMaxScaler()
inputs = scaler.fit_transform(inputs)

outputs = df1['Outcome'].values
#print(outputs.shape)

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.2)

y_train = torch.tensor(y_train, dtype=torch.float)
x_train = torch.tensor(x_train, dtype=torch.float)
dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

network = nn.Sequential(nn.Linear(in_features=8, out_features=5), nn.Sigmoid(), nn.Linear(5,5), nn.Sigmoid(), nn.Linear(5,1), nn.Sigmoid())

loss_function = nn.BCELoss()
optmizer = torch.optim.Adam(network.parameters(), lr = 0.0001)

epochs = 10000
loss_history = np.zeros(epochs)
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
    loss_history[epoch]=running_loss/len(train_loader)
    print('Epoch:' + str(epoch+1) + ' -- loss:' + str(running_loss/len(train_loader)))

print(network.eval())
x_test = torch.tensor(x_test, dtype=torch.float)
predictions = network.forward(x_test)
predictions = np.array(predictions > 0.5)

print('accuracy', accuracy_score(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

mapplot=sns.heatmap(cm, annot=True)
plt.show()

plt.plot(loss_history)
plt.show()
