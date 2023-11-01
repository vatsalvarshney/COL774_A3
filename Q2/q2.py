import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

def get_data(x_path, y_path):
    '''
    Args:
        x_path: path to x file
        y_path: path to y file
    Returns:
        x: np array of [NUM_OF_SAMPLES x n]
        y: np array of [NUM_OF_SAMPLES]
    '''
    x = np.load(x_path)
    y = np.load(y_path)

    y = y.astype('float')
    x = x.astype('float')

    #normalize x:
    x = 2*(0.5 - x/255)
    return x, y

X_train, y_train = get_data('Q2/x_train.npy', 'Q2/y_train.npy')
X_test, y_test = get_data('Q2/x_test.npy', 'Q2/y_test.npy')

label_encoder = OneHotEncoder(sparse_output = False)
label_encoder.fit(np.expand_dims(y_train, axis = -1))

y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis = -1))
y_test_onehot = label_encoder.transform(np.expand_dims(y_test, axis = -1))

# (a)

class NeuralNetwork:
    def __init__(self, M, n, hidden_arch, r, lr=0.01, lr_type=0, activation = 'sigmoid'):
        '''
        Args:
            M: batch size
            n: number of features
            hidden_layer_architecture: list of number of perceptrons in each hidden layer
            r: number of target classes
            lr: learning rate
            lr_type: 0 for constant learning rate, 1 for learning rate decay
            activation: activation function to use in hidden layers ('sigmoid'/'relu'/'leaky_relu')
        '''
        self.M = M
        self.n = n
        self.hidden_arch = hidden_arch
        self.r = r
        self.lr = lr
        self.lr_type = lr_type
        self.activation = activation

        #initialize weights and biases
        self.weights = []
        self.biases = []

        if activation == 'relu' or activation == 'leaky_relu':
            for i in range(len(hidden_arch)):
                if i == 0:
                    self.weights.append(np.random.normal(0, 1/np.sqrt(n), (n, hidden_arch[i])))
                    self.biases.append(np.random.normal(0, 1/np.sqrt(n), (1, hidden_arch[i])))
                else:
                    self.weights.append(np.random.normal(0, 1/np.sqrt(hidden_arch[i-1]), (hidden_arch[i-1], hidden_arch[i])))
                    self.biases.append(np.random.normal(0, 1/np.sqrt(hidden_arch[i-1]), (1, hidden_arch[i])))

            self.weights.append(np.random.normal(0, 1/np.sqrt(hidden_arch[-1]), (hidden_arch[-1], r)))
            self.biases.append(np.random.normal(0, 1/np.sqrt(hidden_arch[-1]), (1, r)))

        else:
            for i in range(len(hidden_arch)):
                if i == 0:
                    self.weights.append(np.random.normal(0, 1, (n, hidden_arch[i])))
                    self.biases.append(np.random.normal(0, 1, (1, hidden_arch[i])))
                else:
                    self.weights.append(np.random.normal(0, 1, (hidden_arch[i-1], hidden_arch[i])))
                    self.biases.append(np.random.normal(0, 1, (1, hidden_arch[i])))

            self.weights.append(np.random.normal(0, 1, (hidden_arch[-1], r)))
            self.biases.append(np.random.normal(0, 1, (1, r)))

        #initialize activations and net values
        self.activations = []
        self.net = []
        for i in range(len(hidden_arch)):
            self.activations.append(np.zeros((M, hidden_arch[i])))
            self.net.append(np.zeros((M, hidden_arch[i])))
        self.activations.append(np.zeros((M, r)))
        self.net.append(np.zeros((M, r)))

        #initialize gradients (partial derivative of loss w.r.t weights)
        self.gradients = []
        for i in range(len(hidden_arch)):
            self.gradients.append(np.zeros((M, hidden_arch[i])))
        self.gradients.append(np.zeros((M, r)))

        #initialize delta (partial derivative of loss w.r.t net values)
        self.delta = []
        for i in range(len(hidden_arch)):
            self.delta.append(np.zeros((M, hidden_arch[i])))
        self.delta.append(np.zeros((M, r)))


    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def softmax(self, x):
        # to avoid overflow
        x -= np.max(x, axis = 1, keepdims = True)
        return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def leaky_relu(self, x):
        return np.maximum(1e-4*x, x)
    

    def forward(self, x:np.ndarray):
        '''
        Args:
            x: np array of [M x n]
        '''
        #forward pass for hidden layers
        for i in range(len(self.hidden_arch)):
            if i == 0:
                self.net[i] = x @ self.weights[i] + self.biases[i]
            else:
                self.net[i] = self.activations[i-1] @ self.weights[i] + self.biases[i]
            if self.activation == 'sigmoid':
                self.activations[i] = self.sigmoid(self.net[i])
            elif self.activation == 'relu':
                self.activations[i] = self.relu(self.net[i])
            elif self.activation == 'leaky_relu':
                self.activations[i] = self.leaky_relu(self.net[i])

        #forward pass for output layer
        self.net[-1] = self.activations[-2] @ self.weights[-1] + self.biases[-1]
        self.activations[-1] = self.softmax(self.net[-1])


    def backward(self, x:np.ndarray, y:np.ndarray):
        '''
        Args:
            x: np array of [M x n]
            y: np array of [M x r]
        '''
        #backward pass for output layer
        self.delta[-1] = (self.activations[-1] - y)
        self.gradients[-1] = (self.activations[-2].T @ self.delta[-1])

        #backward pass for hidden layers
        for i in range(len(self.hidden_arch)-1, 0, -1):
            if self.activation == 'sigmoid':
                self.delta[i] = (self.delta[i+1] @ self.weights[i+1].T) * self.activations[i] * (1 - self.activations[i])
            elif self.activation == 'relu':
                self.delta[i] = (self.delta[i+1] @ self.weights[i+1].T) * (self.net[i] > 0)
            elif self.activation == 'leaky_relu':
                self.delta[i] = (self.delta[i+1] @ self.weights[i+1].T) * (self.net[i] > 0) + 0.01*(self.delta[i+1] @ self.weights[i+1].T) * (self.net[i] < 0)
            self.gradients[i] = self.activations[i-1].T @ self.delta[i]

        #backward pass for first hidden layer
        if self.activation == 'sigmoid':
            self.delta[0] = (self.delta[1] @ self.weights[1].T) * self.activations[0] * (1 - self.activations[0])
        elif self.activation == 'relu':
            self.delta[0] = (self.delta[1] @ self.weights[1].T) * (self.net[0] > 0)
        elif self.activation == 'leaky_relu':
            self.delta[0] = (self.delta[1] @ self.weights[1].T) * (self.net[0] > 0) + 0.01*(self.delta[1] @ self.weights[1].T) * (self.net[0] < 0)
        self.gradients[0] = x.T @ self.delta[0]


    def update(self, epoch=1):
        '''
        Args:
            epoch: current epoch number (1-indexed)
        '''
        if self.lr_type == 0:
            epoch = 1
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr/np.sqrt(epoch) * self.gradients[i]
            self.biases[i] -= self.lr/np.sqrt(epoch) * np.sum(self.delta[i], axis = 0, keepdims = True)


    def fit(self, X_train:np.ndarray, y_train:np.ndarray, max_epochs = 1000, tol = 0, verbose = True):
        '''
        Args:
            X_train: np array of [NUM_OF_SAMPLES x n]
            y_train: np array of [NUM_OF_SAMPLES]
            max_epochs: maximum number of epochs to train
            tol: tolerance for convergence
        '''
        self.loss = []
        m = X_train.shape[0]
        y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis = -1))

        rng = range(max_epochs)
        if verbose:
            rng= tqdm(rng)
        for epoch in rng:
            perm = np.random.permutation(m)
            X_train = X_train[perm]
            y_train_onehot = y_train_onehot[perm]

            for i in range(0, m, self.M):
                x = X_train[i:i+self.M]
                y = y_train_onehot[i:i+self.M]

                self.forward(x)
                self.backward(x, y)
                self.update(epoch+1)

            if tol>0:
                self.forward(X_train)
                loss = -np.sum(y_train_onehot * np.log(self.activations[-1]))/self.M
                self.loss.append(loss)
                if epoch > 0 and abs(self.loss[-1] - self.loss[-2]) < tol:
                    break


    def predict(self, X_test:np.ndarray):
        '''
        Args:
            X_test: np array of [NUM_OF_TEST_SAMPLES x n]
        '''
        self.forward(X_test)
        return 1+np.argmax(self.activations[-1], axis = 1)

# (b)

hidden_archs = [[1], [5], [10], [50], [100]]
precision_scores_train = []
precision_scores_test = []
recall_scores_train = []
recall_scores_test = []
f1_scores_train = []
f1_scores_test = []
for hidden_arch in hidden_archs:
    nn = NeuralNetwork(M = 32, n = 1024, hidden_arch = hidden_arch, r = 5, lr = 0.01)
    nn.fit(X_train, y_train, max_epochs = 1000, tol = 5e-3)
    y_pred_train = nn.predict(X_train)
    cr_train = classification_report(y_pred_train, y_train, output_dict = True, zero_division=0)
    precision_scores_train.append(cr_train['macro avg']['precision'])
    recall_scores_train.append(cr_train['macro avg']['recall'])
    f1_scores_train.append(cr_train['macro avg']['f1-score'])
    y_pred_test = nn.predict(X_test)
    cr_test = classification_report(y_pred_test, y_test, output_dict = True, zero_division=0)
    precision_scores_test.append(cr_test['macro avg']['precision'])
    recall_scores_test.append(cr_test['macro avg']['recall'])
    f1_scores_test.append(cr_test['macro avg']['f1-score'])

df = pd.DataFrame({
    'Hidden Layer Architecture': hidden_archs,
    'Train Precision': precision_scores_train,
    'Test Precision': precision_scores_test,
    'Train Recall': recall_scores_train,
    'Test Recall': recall_scores_test,
    'Train F1': f1_scores_train,
    'Test F1': f1_scores_test
}).round(4)
print(df)

units = [x[0] for x in hidden_archs]
plt.clf()
plt.plot(units, f1_scores_train, label = 'Train')
plt.plot(units, f1_scores_test, label = 'Test')
plt.scatter(units, f1_scores_train)
plt.scatter(units, f1_scores_test)
plt.xlabel('Number of Hidden Units')
plt.ylabel('F1 Score')
plt.xscale('log')
plt.legend()
plt.title('Q2(b): F1 Score vs Number of Hidden Layer Units')
plt.savefig('plots/q2b_f1.png')

# (c)

hidden_archs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
hidden_depths = [1, 2, 3, 4]
precision_scores_train = []
precision_scores_test = []
recall_scores_train = []
recall_scores_test = []
f1_scores_train = []
f1_scores_test = []

for hidden_arch in hidden_archs:
    nn = NeuralNetwork(M = 32, n = 1024, hidden_arch = hidden_arch, r = 5, lr = 0.01)
    nn.fit(X_train, y_train, max_epochs = 200, tol = 1e-3)
    y_pred_train = nn.predict(X_train)
    cr_train = classification_report(y_pred_train, y_train, output_dict = True, zero_division=0)
    precision_scores_train.append(cr_train['macro avg']['precision'])
    recall_scores_train.append(cr_train['macro avg']['recall'])
    f1_scores_train.append(cr_train['macro avg']['f1-score'])
    y_pred_test = nn.predict(X_test)
    cr_test = classification_report(y_pred_test, y_test, output_dict = True, zero_division=0)
    precision_scores_test.append(cr_test['macro avg']['precision'])
    recall_scores_test.append(cr_test['macro avg']['recall'])
    f1_scores_test.append(cr_test['macro avg']['f1-score'])

df = pd.DataFrame({
    'Hidden Layer Architecture': hidden_archs,
    'Train Precision': precision_scores_train,
    'Test Precision': precision_scores_test,
    'Train Recall': recall_scores_train,
    'Test Recall': recall_scores_test,
    'Train F1': f1_scores_train,
    'Test F1': f1_scores_test
}).round(4)
print(df)

plt.clf()
plt.plot(hidden_depths, f1_scores_train, label = 'Train')
plt.plot(hidden_depths, f1_scores_test, label = 'Test')
plt.scatter(hidden_depths, f1_scores_train)
plt.scatter(hidden_depths, f1_scores_test)
plt.xlabel('Network Depth')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Q2(c): F1 Score vs Network Depth')
plt.savefig('plots/q2c_f1.png')

# (d)

hidden_archs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
hidden_depths = [1, 2, 3, 4]
precision_scores_train = []
precision_scores_test = []
recall_scores_train = []
recall_scores_test = []
f1_scores_train = []
f1_scores_test = []

for hidden_arch in hidden_archs:
    nn = NeuralNetwork(M = 32, n = 1024, hidden_arch = hidden_arch, r = 5, lr = 0.01, lr_type = 1)
    nn.fit(X_train, y_train, max_epochs = 200, tol = 1e-4)
    y_pred_train = nn.predict(X_train)
    cr_train = classification_report(y_pred_train, y_train, output_dict = True, zero_division=0)
    precision_scores_train.append(cr_train['macro avg']['precision'])
    recall_scores_train.append(cr_train['macro avg']['recall'])
    f1_scores_train.append(cr_train['macro avg']['f1-score'])
    y_pred_test = nn.predict(X_test)
    cr_test = classification_report(y_pred_test, y_test, output_dict = True, zero_division=0)
    precision_scores_test.append(cr_test['macro avg']['precision'])
    recall_scores_test.append(cr_test['macro avg']['recall'])
    f1_scores_test.append(cr_test['macro avg']['f1-score'])

df = pd.DataFrame({
    'Hidden Layer Architecture': hidden_archs,
    'Train Precision': precision_scores_train,
    'Test Precision': precision_scores_test,
    'Train Recall': recall_scores_train,
    'Test Recall': recall_scores_test,
    'Train F1': f1_scores_train,
    'Test F1': f1_scores_test
}).round(4)
print(df)

plt.clf()
plt.plot(hidden_depths, f1_scores_train, label = 'Train')
plt.plot(hidden_depths, f1_scores_test, label = 'Test')
plt.scatter(hidden_depths, f1_scores_train)
plt.scatter(hidden_depths, f1_scores_test)
plt.xlabel('Network Depth')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Q2(d): F1 Score vs Network Depth')
plt.savefig('plots/q2d_f1.png')

# (e)

hidden_archs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
hidden_depths = [1, 2, 3, 4]
precision_scores_train = []
precision_scores_test = []
recall_scores_train = []
recall_scores_test = []
f1_scores_train = []
f1_scores_test = []

for hidden_arch in hidden_archs:
    nn = NeuralNetwork(M = 32, n = 1024, hidden_arch = hidden_arch, r = 5, lr = 0.01, lr_type = 1, activation = 'leaky_relu')
    nn.fit(X_train, y_train, max_epochs = 100, tol = 0)
    y_pred_train = nn.predict(X_train)
    cr_train = classification_report(y_pred_train, y_train, output_dict = True, zero_division=0)
    precision_scores_train.append(cr_train['macro avg']['precision'])
    recall_scores_train.append(cr_train['macro avg']['recall'])
    f1_scores_train.append(cr_train['macro avg']['f1-score'])
    print(cr_train['macro avg']['f1-score'])
    y_pred_test = nn.predict(X_test)
    cr_test = classification_report(y_pred_test, y_test, output_dict = True, zero_division=0)
    precision_scores_test.append(cr_test['macro avg']['precision'])
    recall_scores_test.append(cr_test['macro avg']['recall'])
    f1_scores_test.append(cr_test['macro avg']['f1-score'])

df = pd.DataFrame({
    'Hidden Layer Architecture': hidden_archs,
    'Train Precision': precision_scores_train,
    'Test Precision': precision_scores_test,
    'Train Recall': recall_scores_train,
    'Test Recall': recall_scores_test,
    'Train F1': f1_scores_train,
    'Test F1': f1_scores_test
}).round(4)
print(df)

plt.clf()
plt.plot(hidden_depths, f1_scores_train, label = 'Train')
plt.plot(hidden_depths, f1_scores_test, label = 'Test')
plt.scatter(hidden_depths, f1_scores_train)
plt.scatter(hidden_depths, f1_scores_test)
plt.xlabel('Network Depth')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Q2(e): F1 Score vs Network Depth')
plt.savefig('plots/q2e_f1.png')

# (f)

hidden_archs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
hidden_depths = [1, 2, 3, 4]
precision_scores_train = []
precision_scores_test = []
recall_scores_train = []
recall_scores_test = []
f1_scores_train = []
f1_scores_test = []

for hidden_arch in hidden_archs:
    nn = MLPClassifier(hidden_layer_sizes=hidden_arch, activation='relu', solver='sgd', alpha=0, batch_size=32, learning_rate='invscaling')
    nn.fit(X_train, y_train)
    y_pred_train = nn.predict(X_train)
    cr_train = classification_report(y_pred_train, y_train, output_dict = True, zero_division=0)
    precision_scores_train.append(cr_train['macro avg']['precision'])
    recall_scores_train.append(cr_train['macro avg']['recall'])
    f1_scores_train.append(cr_train['macro avg']['f1-score'])
    y_pred_test = nn.predict(X_test)
    cr_test = classification_report(y_pred_test, y_test, output_dict = True, zero_division=0)
    precision_scores_test.append(cr_test['macro avg']['precision'])
    recall_scores_test.append(cr_test['macro avg']['recall'])
    f1_scores_test.append(cr_test['macro avg']['f1-score'])

df = pd.DataFrame({
    'Hidden Layer Architecture': hidden_archs,
    'Train Precision': precision_scores_train,
    'Test Precision': precision_scores_test,
    'Train Recall': recall_scores_train,
    'Test Recall': recall_scores_test,
    'Train F1': f1_scores_train,
    'Test F1': f1_scores_test
}).round(4)
print(df)

plt.clf()
plt.plot(hidden_depths, f1_scores_train, label = 'Train')
plt.plot(hidden_depths, f1_scores_test, label = 'Test')
plt.scatter(hidden_depths, f1_scores_train)
plt.scatter(hidden_depths, f1_scores_test)
plt.xlabel('Network Depth')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Q2(f): F1 Score vs Network Depth')
plt.savefig('plots/q2f_f1.png')


