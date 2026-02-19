import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

'''
General Instructions:

1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
the library above available.

2. You are expected to fill in the skeleton code precisely as per provided. On top of skeleton code given,
you may write whatever deemed necessary to complete the assignment. For example, you may define additional 
default arguments, class parameters, or methods to help you complete the assignment.

3. Some initial steps or definition are given, aiming to help you getting started. As long as you follow 
the argument and return type, you are free to change them as you see fit.

4. Your code should be free of compilation errors. Compilation errors will result in 0 marks.
'''


'''
Problem A-1: Data Preprocessing and EDA
'''
class DataLoader:
    '''
    This class will be used to load the data and perform initial data processing. Fill in functions.
    You are allowed to add any additional functions which you may need to check the data. This class 
    will be tested on the pre-built enviornment with only numpy and pandas available.
    '''

    def __init__(self, data_root: str, random_state: int = 42):
        '''
        Inialize the DataLoader class with the data_root path.
        Load data as pd.DataFrame, store as needed and initialize other variables.
        All dataset should save as pd.DataFrame.
        '''
        self.random_state = random_state
        np.random.seed(self.random_state)

        print("random state: ", random_state)

        # Load dataset
        data_path = data_root.rstrip("/") + "/hw2-bank_data.csv"
        self.data = pd.read_csv(data_path, sep=';')

        self.data_train = None
        self.data_valid = None

        # Call class methods
        print("=========================== INIT START =================================")
        self.data_prep()
        self.data_split()
        self.extract_features_and_label()
        print("=========================== INIT STOP =================================")
        

    def data_split(self) -> None:
        '''
        You are asked to split the training data into train/valid datasets on the ratio of 80/20. 
        Add the split datasets to self.data_train, self.data_valid. Both of the split should still be pd.DataFrame.
        '''
        if self.data is None or len(self.data) == 0:
            # Keep consistent types even if empty
            self.data_train = pd.DataFrame()
            self.data_valid = pd.DataFrame()
            return

        n = len(self.data)
        n_train = int(0.8 * n)

        # Shuffle indices
        indices = np.arange(n)
        np.random.shuffle(indices)

        train_idx = indices[:n_train]
        valid_idx = indices[n_train:]

        # Use iloc because indices are positional
        self.data_train = self.data.iloc[train_idx].reset_index(drop=True)
        self.data_valid = self.data.iloc[valid_idx].reset_index(drop=True)

    def data_prep(self) -> None:
        '''
        You are asked to drop any rows with missing values and map categorical variables to numeric values. 
        '''
        if self.data is None or len(self.data) == 0:
            self.data = pd.DataFrame()
            return

        # 1) Drop rows with missing values
        self.data = self.data.dropna().reset_index(drop=True)

        # Normalize all object columns first (strip spaces and quotes)
        obj_cols = self.data.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            self.data[col] = self.data[col].astype(str).str.strip().str.strip('"').str.strip("'")

        # Explicit yes/no mapping for known binary columns (including label)
        yn_map = {"no": 0, "yes": 1}
        for col in ["default", "housing", "loan", "y"]:
            if col in self.data.columns:
                # If already numeric, leave it alone (prevents turning 0/1 into NaN)
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    continue
                # Otherwise map normalized strings
                self.data[col] = self.data[col].map(yn_map)

        # For remaining object columns, use pandas categorical codes
        obj_cols = self.data.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            self.data[col] = pd.Categorical(self.data[col]).codes

    def extract_features_and_label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''
        This function will be called multiple times to extract features and labels from train/valid/test 
        data.
        
        Expected return:
            X_data: np.ndarray of shape (n_samples, n_features) - Extracted features
            y_data: np.ndarray of shape (n_samples,) - Extracted labels
        '''
        # Features: everything except label column "y"
        X_data = data.drop(columns=["y"]).to_numpy()

        # Label: column "y"
        y_data = data["y"].to_numpy()

        print("Features and Labels")
        print("X_data: ", X_data)
        print("y_data: ", y_data)

        return X_data, y_data

    def plot_histograms(self, data: pd.DataFrame) -> None:
        '''
        Plot histograms for all columns in self.data.
        
        '''
        if data is None or len(data) == 0:
            return

        cols = list(data.columns)
        n_cols = 4
        n_rows = int(np.ceil(len(cols) / n_cols))

        plt.figure(figsize=(4 * n_cols, 3 * n_rows))
        for i, col in enumerate(cols, start=1):
            plt.subplot(n_rows, n_cols, i)
            plt.hist(data[col].to_numpy(), bins=20, edgecolor="black")
            plt.title(str(col))
            plt.tight_layout()

'''
Porblem A-2: Classification Tree Inplementation
'''
class ClassificationTree:
    '''
    You are asked to implement a simple classification tree from scratch. This class will be tested on the
    pre-built enviornment with only numpy and pandas available.

    You may add more variables and functions to this class as you see fit.
    '''
    class Node:
        '''
        A data structure to represent a node in the tree.
        '''
        def __init__(self, split=None, left=None, right=None, prediction=None):
            '''
            split: tuple - (feature_idx, split_value)
            left: Node - Left child node
            right: Node - Right child node
            prediction: (any) - Prediction value if the node is a leaf
            '''
            self.split = split
            self.left = left
            self.right = right
            self.prediction = prediction 

        def is_leaf(self):
            return self.prediction is not None

    def __init__(self, random_state: int, max_depth: int = 5):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_depth = max_depth

        self.tree_root = None

    def split_crit(self, y: np.ndarray) -> float:
        '''
        Implement the impurity measure of your choice here. Return the impurity value.
        '''
        # Gini impurity
        n = y.shape[0]
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / n
        return 1.0 - np.sum(p ** 2)        

    def _majority_class(self, y: np.ndarray):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]
  

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> None:
        '''
        Implement the tree building algorithm here. You can recursivly call this function to build the 
        tree. After building the tree, store the root node in self.tree_root.

        Think about the difference between depth here and the max_depth parameter in the constructor.
        '''
        # stopping conditions
        if X.shape[0] == 0:
            return self.Node(prediction=None)  # should not usually happen

        if depth >= self.max_depth:
            return self.Node(prediction=self._majority_class(y))

        if np.unique(y).shape[0] == 1:
            return self.Node(prediction=y[0])

        split = self.search_best_split(X, y)
        if split is None:
            return self.Node(prediction=self._majority_class(y))

        feat_idx, thr = split
        xj = X[:, feat_idx]
        left_mask = xj <= thr
        right_mask = ~left_mask

        # if split degenerates, make leaf
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self.Node(prediction=self._majority_class(y))

        left_node = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        return self.Node(split=(feat_idx, thr), left=left_node, right=right_node, prediction=None)


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Fit the classification tree to the training data. This is the function that will be called to train your model in autograder.
        
        Args:
            X: numpy array of shape (n_samples, n_features) - Training features
            y: numpy array of shape (n_samples,) - Training labels
        '''
        self.tree_root = self.build_tree(X, y, depth=0)

    def search_best_split(self, X: np.ndarray, y: np.ndarray):
        '''
        Implement the search for best split here.

        Expected return:
        - tuple(int, float): Best feature index and split value
        - None: If no split is found
        '''
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict classes for multiple samples.
        
        Args:
            X: numpy array with the same columns as the training data
            
        Returns:
            np.ndarray: Array of predictions
        '''
        pass


def train_XGBoost() -> dict:
    '''
    See instruction for implementation details. This function will be tested on the pre-built enviornment
    with numpy, pandas, xgboost available.
    '''
    pass


'''
Initialize the following variable with the best model you have found. This model will be used in testing 
in our pre-built environment.
'''
my_best_model = XGBClassifier()


if __name__ == "__main__":
    dataLoader = DataLoader(data_root="./", random_state=42)
    