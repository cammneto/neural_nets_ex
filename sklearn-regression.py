import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

boston = datasets.load_boston()
inputs = boston.data
outputs = boston.target

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.2)

scaler1 = MinMaxScaler()
X_train = scaler1.fit_transform(X_train)
X_test = scaler1.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
scaler2 = MinMaxScaler()
y_train = scaler2.fit_transform(y_train)
y_test = scaler2.transform(y_test)

network = MLPRegressor(max_iter=2000, verbose=True, hidden_layer_sizes=(7))
network.fit(X_train, y_train)

predictions = network.predict(X_test)
print('predictions',predictions)
print('y_test',y_test)

print('mean absolut error',mean_absolute_error(y_test, predictions))
print('mean squared error',np.sqrt(mean_squared_error(y_test, predictions)))

