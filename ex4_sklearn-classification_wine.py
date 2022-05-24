from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from yellowbrick.classifier import ConfusionMatrix

wines =  datasets.load_wine()
inputs = wines.data
outputs = wines.target

print(outputs.shape)
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.2)

network = MLPClassifier(max_iter=2000, verbose=True, tol=0.000001,
                        activation = 'logistic', solver = 'adam',
                        learning_rate = 'constant', learning_rate_init = 0.002,
                        batch_size = 32, hidden_layer_sizes = (8, 8)
                        #early_stopping = True, #n_iter_no_change = 50
                        )
network.fit(X_train, y_train)
predictions = network.predict(X_test)
print(accuracy_score(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
confusion_matrix = ConfusionMatrix(network, classes = wines.target_names)
confusion_matrix.fit(X_train, y_train)
confusion_matrix.score(X_test, y_test)
confusion_matrix.show()
