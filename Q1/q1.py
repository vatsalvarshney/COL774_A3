from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from time import time


# (a)

label_encoder = None 

def get_np_array_ordinal(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OrdinalEncoder()
        # label_encoder = OrdinalEncoder(sparse_output = False)
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()

def get_np_array_one_hot(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OneHotEncoder(sparse_output = False)
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()


class DTNode:
    # Decision Tree Node

    def __init__(self, depth):
        self.depth = depth
        self.value = 0
        self.is_leaf = False
        self.column = None
        self.children = {}
        self.type = None
        self.median = None


    def get_children(self, X):
        '''
        Args:
            X: A single example np array [num_features]
        Returns:
            child: A DTNode
        '''
        if self.is_leaf:
            return None
        if self.type == 'cat':
            if X[self.column] in self.children:
                return self.children[X[self.column]]
            else:
                return None
        if self.type == 'cont':
            if X[self.column] <= self.median:
                return self.children[0]
            else:
                return self.children[1]
        return None


class DTTree:

    def __init__(self):
        self.root = None


    def fit(self, X, y, types, max_depth = 10):
        '''
        Makes decision tree

        Args:
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
            types: list of [num_features] with types as: cat, cont
                eg: if num_features = 4, and last 2 features are continious then
                    types = ['cat','cat','cont','cont']
            max_depth: maximum depth of tree
        Returns:
            None
        '''
        self.root = DTNode(0)
        self.root = self.__fit(self.root, X, y, types, max_depth)
        return None


    def __fit(self, node: DTNode, X: np.ndarray, y: np.ndarray, types: list, max_depth: int) -> DTNode:
        '''
        Recursively makes decision tree

        Args:
            node: DTNode
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
            types: list of [num_features] with types as: cat, cont
                eg: if num_features = 4, and last 2 features are continious then
                    types = ['cat','cat','cont','cont']
            max_depth: maximum depth of tree
        Returns:
            node: DTNode
        '''

        node.value = stats.mode(y)[0][0]

        if node.depth == max_depth or np.unique(y).size == 1:
            node.is_leaf = True
            return node
        
        mutual_info = [self.__mutual_info(X[:,i], y[:,0], types[i]) for i in range(X.shape[1])]
        node.column = np.argmax(mutual_info)
        node.type = types[node.column]
        x_split = X[:,node.column]

        node.children = {}
        if node.type == 'cat':
            for i in np.unique(x_split):
                idx = x_split == i
                node.children[i] = DTNode(node.depth+1)
                node.children[i] = self.__fit(node.children[i], X[idx], y[idx], types, max_depth)
        else:
            node.median = np.median(x_split)
            less = x_split <= node.median
            node.children[0] = DTNode(node.depth+1)
            node.children[0] = self.__fit(node.children[0], X[less], y[less], types, max_depth)
            node.children[1] = DTNode(node.depth+1)
            node.children[1] = self.__fit(node.children[1], X[~less], y[~less], types, max_depth)

        return node


    def __mutual_info(self, x: np.ndarray, y: np.ndarray, type='cat'):
        '''
        Calculates mutual information between x and y

        Args:
            x: numpy array of data [num_samples]
            y: numpy array of classes [num_samples]
            type: type of attribute x, 'cat' (categorical) or 'cont' (continuous)
        Returns:
            mutual_info: float
        '''
        splits = []
        if type == 'cat':
            splits = [x == i for i in np.unique(x)]
        else:
            less = x <= np.median(x)
            splits = [less, ~less]
        mutual_info = 0
        for split in splits:
            mutual_info += np.sum(split) / x.size * self.__entropy(y[split])
        mutual_info = self.__entropy(y) - mutual_info
        return mutual_info

    
    def __entropy(self, y: np.ndarray) -> float:
        '''
        Calculates entropy of y

        Args:
            y: numpy array of classes [num_samples]
        Returns:
            entropy: float
        '''
        entropy = 0
        for i in np.unique(y):
            p = np.sum(y == i) / y.size
            entropy -= p * np.log2(p)
        return entropy


    def __call__(self, X):
        '''
        Predicted classes for X

        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        '''
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            y[i] = self.__pred(self.root, X[i])
        return y


    def __pred(self, node: DTNode, X: np.ndarray):
        '''
        Predicted class for a single example X

        Args:
            node: DTNode
            X: numpy array of data [num_features]
        Returns:
            y: float predicted class
        '''
        child = node.get_children(X)
        if child is None:
            return node.value
        return self.__pred(child, X)


    def accuracy(self, X: np.ndarray, y: np.ndarray):
        '''
        Calculates accuracy of the model on X and y

        Args:
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
        Returns:
            accuracy: float between [0,1]
        '''
        return np.sum(self(X) == y) / y.size


    def get_nodes(self) -> list[DTNode]:
        '''
        Returns all nodes of the tree

        Args:
            None
        Returns:
            nodes: list of DTNode
        '''
        nodes = []
        self.__get_nodes(self.root, nodes)
        return nodes

    def __get_nodes(self, node: DTNode, nodes: list):
        '''
        Recursively gets all nodes of the tree

        Args:
            node: DTNode
            nodes: list of DTNode
        Returns:
            None
        '''
        nodes.append(node)
        if not node.is_leaf:
            for child in node.children.values():
                self.__get_nodes(child, nodes)
    

    def post_prune(self, X_train: np.ndarray, y_train:np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray):
        '''
        Post prunes the tree

        Args:
            X_train: numpy array of data [num_samples, num_features]
            y_train: numpy array of classes [num_samples, 1]
            X_val: numpy array of data [num_samples, num_features]
            y_val: numpy array of classes [num_samples, 1]
            X_test: numpy array of data [num_samples, num_features]
            y_test: numpy array of classes [num_samples, 1]
        Returns:
            num_nodes: numpy array of number of nodes [num_nodes]
            train_accs: numpy array of accuracies [num_nodes + 1]
            val_accs: numpy array of accuracies [num_nodes + 1]
            test_accs: numpy array of accuracies [num_nodes + 1]
        '''
        
        num_nodes = []
        train_accs = []
        val_accs = []
        test_accs = []

        while True:
            nodes = self.get_nodes()
            num_nodes.append(len(nodes))
            acc = self.accuracy(X_val, y_val)
            train_accs.append(self.accuracy(X_train, y_train))
            val_accs.append(acc)
            test_accs.append(self.accuracy(X_test, y_test))
            max_acc = 0
            max_node = None
            for node in nodes:
                if node.is_leaf:
                    continue
                node.is_leaf = True
                acc_new = self.accuracy(X_val, y_val)
                if acc_new > max_acc:
                    max_acc = acc_new
                    max_node = node
                node.is_leaf = False
            if max_acc > acc:
                max_node.is_leaf = True
            else:
                break
        
        return np.array(num_nodes), np.array(train_accs), np.array(val_accs), np.array(test_accs)


X_train, y_train = get_np_array_ordinal('Q1/train.csv')
X_test, y_test = get_np_array_ordinal('Q1/test.csv')
types = ['cat','cat','cat','cat','cat','cont','cat','cat','cat','cont','cont','cont']


depths = [5, 10, 15, 20, 25]
train_acc = []
test_acc = []
for depth in tqdm(depths):
    tree = DTTree()
    tree.fit(X_train, y_train, types, depth)
    train_acc.append(100*tree.accuracy(X_train, y_train))
    test_acc.append(100*tree.accuracy(X_test, y_test))


df = pd.DataFrame({'Depth':depths, 'Train accuracy (%)':train_acc, 'Test accuracy (%)':test_acc}).round(4)
print(df)
only_win_acc = 100*np.sum(y_test == 1) / y_test.size
only_lose_acc = 100*np.sum(y_test == 0) / y_test.size
print(f'Only win accuracy: {only_win_acc:.4f} %')
print(f'Only lose accuracy: {only_lose_acc:.4f} %')


plt.plot(depths, train_acc, label = 'Train accuracy')
plt.plot(depths, test_acc, label = 'Test accuracy')
plt.scatter(depths, train_acc)
plt.scatter(depths, test_acc)
plt.xlabel('Maximum depth')
plt.ylabel('Accuracy (%)')
plt.title('Q1(a): Accuracy vs Maximum depth (Ordinal Encoding)')
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig('plots/q1a.png')


# (b)


label_encoder = None
X_train, y_train = get_np_array_one_hot('Q1/train.csv')
X_test, y_test = get_np_array_one_hot('Q1/test.csv')
X_val, y_val = get_np_array_one_hot('Q1/val.csv')
types = ['cat','cat','cat','cat','cat','cont','cat','cat','cat','cont','cont','cont']
while len(types) != X_train.shape[1]:
    types = ['cat'] + types


depths = [15,25,29,35,45,60,75]
trees = []
train_acc = []
test_acc = []
for depth in tqdm(depths):
    trees.append(DTTree())
    trees[-1].fit(X_train, y_train, types, depth)
    train_acc.append(100*trees[-1].accuracy(X_train, y_train))
    test_acc.append(100*trees[-1].accuracy(X_test, y_test))


df = pd.DataFrame({'Depth':depths, 'Train accuracy (%)':train_acc, 'Test accuracy (%)':test_acc}).round(4)
print(df)


plt.clf()
plt.plot(depths, train_acc, label = 'Train accuracy')
plt.plot(depths, test_acc, label = 'Test accuracy')
plt.scatter(depths, train_acc)
plt.scatter(depths, test_acc)
plt.xlabel('Maximum depth')
plt.ylabel('Accuracy (%)')
plt.title('Q1(b): Accuracy vs Maximum depth (One Hot Encoding)')
plt.legend()
plt.savefig('plots/q1b.png')


# (c)


for tree,dep in [(trees[0],15),(trees[1],25),(trees[3],35),(trees[4],45)]:
    print(f'Pruning tree with max depth = {dep}... ', end='')
    st = time()
    num_nodes, train_acc, val_acc, test_acc = tree.post_prune(X_train, y_train, X_val, y_val, X_test, y_test)
    print(f'Done in {time()-st:.2f} s')
    plt.clf()
    plt.plot(num_nodes, train_acc, label = 'Train')
    plt.plot(num_nodes, val_acc, label = 'Validation')
    plt.plot(num_nodes, test_acc, label = 'Test')
    plt.scatter(num_nodes, train_acc, s=10)
    plt.scatter(num_nodes, val_acc, s=10)
    plt.scatter(num_nodes, test_acc, s=10)
    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Q1(c): Accuracy vs Number of nodes (max depth = {dep})')
    plt.legend()
    plt.savefig(f'plots/q1c_{dep}.png')


#  (d)


tree = DecisionTreeClassifier(criterion = 'entropy')
tree.fit(X_train, y_train)
train_acc = 100*tree.score(X_train, y_train)
test_acc = 100*tree.score(X_test, y_test)
print(f'Train accuracy: {train_acc:.4f} %')
print(f'Test accuracy: {test_acc:.4f} %')


depths = [15,25,35,45]
train_acc = []
val_acc = []
test_acc = []
for depth in depths:
    tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = depth)
    tree.fit(X_train, y_train)
    train_acc.append(100*tree.score(X_train, y_train))
    val_acc.append(100*tree.score(X_val, y_val))
    test_acc.append(100*tree.score(X_test, y_test))

df = pd.DataFrame({'Depth':depths, 'Train accuracy (%)':train_acc, 'Validation accuracy (%)':val_acc, 'Test accuracy (%)':test_acc}).round(4)
print(df)

plt.clf()
plt.plot(depths, train_acc, label = 'Train')
plt.plot(depths, val_acc, label = 'Validation')
plt.plot(depths, test_acc, label = 'Test')
plt.scatter(depths, train_acc)
plt.scatter(depths, val_acc)
plt.scatter(depths, test_acc)
plt.xlabel('Maximum depth')
plt.ylabel('Accuracy (%)')
plt.title('Q1(d): Accuracy vs Maximum depth (Sklearn)')
plt.legend()
plt.savefig('plots/q1d_depth.png')


ccp_alphas = [0.001, 0.01, 0.1, 0.2]
train_acc = []
val_acc = []
test_acc = []

for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(criterion = 'entropy', ccp_alpha = ccp_alpha)
    tree.fit(X_train, y_train)
    train_acc.append(100*tree.score(X_train, y_train))
    val_acc.append(100*tree.score(X_val, y_val))
    test_acc.append(100*tree.score(X_test, y_test))

df = pd.DataFrame({'ccp_alpha':ccp_alphas, 'Train accuracy (%)':train_acc, 'Validation accuracy (%)':val_acc, 'Test accuracy (%)':test_acc}).round(4)
print(df)

plt.clf()
plt.plot(ccp_alphas, train_acc, label = 'Train')
plt.plot(ccp_alphas, val_acc, label = 'Validation')
plt.plot(ccp_alphas, test_acc, label = 'Test')
plt.scatter(ccp_alphas, train_acc)
plt.scatter(ccp_alphas, val_acc)
plt.scatter(ccp_alphas, test_acc)
plt.xscale('log')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy (%)')
plt.title('Q1(d): Accuracy vs ccp_alpha (Sklearn)')
plt.legend()
plt.savefig('plots/q1d_ccp.png')


#  (e)


param_grid = {
    'n_estimators': [50, 150, 250, 350],
    'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
    'min_samples_split': [2, 4, 6, 8, 10]
}

X_cv = np.concatenate((X_train, X_val), axis=0)
y_cv = np.concatenate((y_train, y_val), axis=0)
test_fold = np.concatenate((np.zeros(X_train.shape[0]), np.ones(X_val.shape[0])), axis=0)
ps = PredefinedSplit(test_fold-1)
rf = RandomForestClassifier(oob_score = True, criterion='entropy')
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = ps, n_jobs = -1, verbose=3)
grid_search.fit(X_cv, y_cv.ravel())


print(f'Best parameters: {grid_search.best_params_}')
print(f'Best training accuracy: {100*grid_search.best_estimator_.score(X_train, y_train):.4f} %')
print(f'Best out-of-bag accuracy: {100*grid_search.best_estimator_.oob_score_:.4f} %')
print(f'Best validation accuracy: {100*grid_search.best_estimator_.score(X_val, y_val):.4f} %')
print(f'Best test accuracy: {100*grid_search.best_estimator_.score(X_test, y_test):.4f} %')


