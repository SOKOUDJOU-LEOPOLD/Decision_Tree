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
        # print("X_data: ", X_data)
        # print("y_data: ", y_data)

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
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None

        parent_impurity = self.split_crit(y)
        best_impurity = parent_impurity
        best_split = None

        for j in range(n_features):
            xj = X[:, j]
            uniq = np.unique(xj)
            if uniq.shape[0] <= 1:
                continue

            # candidate thresholds: midpoints between sorted unique values
            uniq_sorted = np.sort(uniq)
            thresholds = (uniq_sorted[:-1] + uniq_sorted[1:]) / 2.0

            for thr in thresholds:
                left_mask = xj <= thr
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                if n_left == 0 or n_right == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                impurity = (n_left / n_samples) * self.split_crit(y_left) + (n_right / n_samples) * self.split_crit(y_right)

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_split = (j, float(thr))

        return best_split

    def _predict_one(self, x: np.ndarray):
        node = self.tree_root
        while node is not None and not node.is_leaf():
            feat_idx, thr = node.split
            if x[feat_idx] <= thr:
                node = node.left
            else:
                node = node.right

        # Fallback if something odd happens
        if node is None or node.prediction is None:
            return 0
        return node.prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict classes for multiple samples.
        
        Args:
            X: numpy array with the same columns as the training data
            
        Returns:
            np.ndarray: Array of predictions
        '''
        preds = np.array([self._predict_one(X[i]) for i in range(X.shape[0])])
        return preds

# F1 Score
def f1_score_binary(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return (2 * tp) / denom

def train_XGBoost() -> dict:
    '''
    See instruction for implementation details. This function will be tested on the pre-built enviornment
    with numpy, pandas, xgboost available.
    '''
    dl = DataLoader(data_root="./", random_state=42)
    X_train, y_train = dl.extract_features_and_label(dl.data_train)
    X_valid, y_valid = dl.extract_features_and_label(dl.data_valid)

    alpha_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    n_boot = 100

    # class imbalance weight
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    print("scale_pos_weight: ", scale_pos_weight)

    n_train = X_train.shape[0]
    best_alpha = None
    best_f1 = -1.0

    for a in alpha_vals:
        f1_scores = []
        for _ in range(n_boot):
            idx = np.random.randint(0, n_train, size=n_train)  # bootstrap
            Xb = X_train[idx]
            yb = y_train[idx]

            model = XGBClassifier(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,  # 0.8
                colsample_bytree=0.09,
                reg_alpha=a,  #10
                reg_lambda=1,
                min_child_weight=7, 
                gamma=0.01,
                scale_pos_weight=4, #08235294117647, #7.5
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )
            model.fit(Xb, yb)
            pred = model.predict(X_valid)
            f1_scores.append(f1_score_binary(y_valid, pred, pos_label=1))

        avg_f1 = float(np.mean(f1_scores))
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_alpha = a

    # Train final model on full training data with best alpha
    final_model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,  # 0.8
        colsample_bytree=0.09,
        reg_alpha=best_alpha,  #10
        reg_lambda=1,
        min_child_weight=7, 
        gamma=0.01,
        scale_pos_weight=4, #08235294117647, #7.5
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    final_model.fit(X_train, y_train)

    global my_best_model
    my_best_model = final_model

    return {
        "best_alpha": best_alpha,
        "best_valid_bootstrap_f1": best_f1,
        "model": final_model
    }

def plot_roc_auc(model, X: np.ndarray, y: np.ndarray) -> float:
    """
    Plots ROC curve and returns AUC.
    Assumes binary labels y in {0,1}.
    """
    # scores = P(y=1)
    scores = model.predict_proba(X)[:, 1]

    # thresholds from high to low
    thresholds = np.unique(scores)[::-1]

    P = np.sum(y == 1)
    N = np.sum(y == 0)

    tpr = []
    fpr = []

    for thr in thresholds:
        y_hat = (scores >= thr).astype(int)
        tp = np.sum((y == 1) & (y_hat == 1))
        fp = np.sum((y == 0) & (y_hat == 1))

        tpr.append(tp / P if P > 0 else 0.0)
        fpr.append(fp / N if N > 0 else 0.0)

    # Adding endpoints (0,0) and (1,1) for a good ROC curve
    fpr = np.array([0.0] + fpr + [1.0])
    tpr = np.array([0.0] + tpr + [1.0])

    # AUC via trapezoidal rule
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    auc = float(np.trapezoid(tpr, fpr))

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    return auc


'''
Initialize the following variable with the best model you have found. This model will be used in testing 
in our pre-built environment.
'''
my_best_model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,  # 0.8
        colsample_bytree=0.09,
        reg_alpha=0.01,  #10
        reg_lambda=1,
        min_child_weight=7, 
        gamma=0.01,
        scale_pos_weight=4, #08235294117647, #7.5
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
)


if __name__ == "__main__":
    # Training the model
    dataLoader = DataLoader(data_root="./", random_state=42)
    X_train, y_train = dataLoader.extract_features_and_label(dataLoader.data_train)
    X_valid, y_valid = dataLoader.extract_features_and_label(dataLoader.data_valid)

    tree = ClassificationTree(random_state=42, max_depth=5)
    tree.fit(X_train, y_train)

    y_pred_valid = tree.predict(X_valid)

    valid_acc = np.mean(y_pred_valid == y_valid)
    print("Validation accuracy:", valid_acc)   

    #Valid data F1-score
    print("Valid F1:", f1_score_binary(y_valid, y_pred_valid, pos_label=1))

    # Tuning max-depth to find the best F1 score
    best_f1, best_depth = -1, None
    for d in [1,2,3,4,5,6,8,10]:
        tree = ClassificationTree(random_state=42, max_depth=d)
        tree.fit(X_train, y_train)
        f1 = f1_score_binary(y_valid, tree.predict(X_valid))
        if f1 > best_f1:
            best_f1, best_depth = f1, d
    print("best_depth: ", best_depth," best_f1: ", best_f1)

    # Tuning XGBClassifier
    print(train_XGBoost())

    # Plot ROC Curve
    my_best_model.fit(X_train, y_train)
    auc = plot_roc_auc(my_best_model, X_valid, y_valid)
    plt.show()
    print("AUC:", auc)