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

    def __init__(self, data_root: str, random_state: int):
        '''
        Inialize the DataLoader class with the data_root path.
        Load data as pd.DataFrame, store as needed and initialize other variables.
        All dataset should save as pd.DataFrame.
        '''
        self.random_state = random_state
        np.random.seed(self.random_state)

        # Load dataset
        data_path = data_root.rstrip("/") + "/hw2-bank_data.csv"
        self.data = pd.read_csv(data_path, sep=';')

        self.data_train = None
        self.data_valid = None

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

        # Explicit label mapping
        if "y" in self.data.columns:
            self.data["y"] = self.data["y"].astype(str).map({"no": 0, "yes": 1})


        # 2) Map categorical (object) columns to numeric
        cat_cols = self.data.select_dtypes(include=["object"]).columns
        cat_cols = [c for c in cat_cols if c != "y"]

        for col in cat_cols:
            # make mapping deterministic: categories sorted lexicographically
            cats = sorted(self.data[col].astype(str).unique().tolist())
            self.data[col] = pd.Categorical(self.data[col].astype(str), categories=cats).codes

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
        pass
        
    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> None:
        '''
        Implement the tree building algorithm here. You can recursivly call this function to build the 
        tree. After building the tree, store the root node in self.tree_root.

        Think about the difference between depth here and the max_depth parameter in the constructor.
        '''
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Fit the classification tree to the training data. This is the function that will be called to train your model in autograder.
        
        Args:
            X: numpy array of shape (n_samples, n_features) - Training features
            y: numpy array of shape (n_samples,) - Training labels
        '''
        pass

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
    dataLoader = DataLoader(data_root="./", random_state=0)
    # print training data
    dataLoader.data_split()
    print(dataLoader.data_train)
    # create histogram
    dataLoader.plot_histograms(dataLoader.data_train)
    plt.show()

    # print preprocessed data i.e. numeric values of columns
    dataLoader.data_prep()
    print(dataLoader.data)