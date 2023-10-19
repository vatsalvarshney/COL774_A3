'''
This is started code for part b and c. 
Using this code is OPTIONAL and you may write code from scratch if you want
'''


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np

label_encoder = None 

def get_np_array(file_name):
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

    def __init__(self, depth, is_leaf = False, value = 0, column = None):

        #to split on column
        self.depth = depth

        #add children afterwards
        self.children = None

        #if leaf then also need value
        self.is_leaf = is_leaf
        if(self.is_leaf):
            self.value = value
        
        if(not self.is_leaf):
            self.column = column


    def get_children(self, X):
        '''
        Args:
            X: A single example np array [num_features]
        Returns:
            child: A DTNode
        '''
        #TODO


class DTTree:

    def __init__(self):
        #Tree root should be DTNode
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
        #TODO

    def __call__(self, X):
        '''
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        '''
        #TODO
    
    def post_prune(self, X_val, y_val):
        #TODO
 

if __name__ == '__main__':

    #change the path if you want
    X_train,y_train = get_np_array('train.csv')
    X_test, y_test = get_np_array("test.csv")

    #only needed in part (c)
    X_val, y_val = get_np_array("val.csv")

    types = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont" ]
    while(len(types) != X.shape[1]):
        types = ['cat'] + types

    max_depth = 10
    tree = DTTree()
    tree.fit(X_train,y_train,types, max_depth = max_depth)
    
    #this is only applicable in part c
    tree.post_prune(X_val, y_val)
    