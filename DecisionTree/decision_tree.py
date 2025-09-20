# below implementation of Decision Tree is done using Gini Impurity. An option of ID3 (entropy) would be added soon.
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 100, n_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth = 0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # cheking the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value = leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace = False)

        # find the best split using gini impurity
        best_feature, best_thresh = self.best_split(X, y, feat_idxs)

        # creating child nodes
        left_idxs, right_idxs =  self.split(X[:, best_feature], best_thresh)
        left = self.grow_tree(X[left_idxs, :], y[left_idxs], depth + 1) 
        right = self.grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)
    
    def best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                # calculate gini
                gain = self.gini_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
    
    def gini_gain(self, y, X_column, threshold):
        # parent gini
        parent_gini = self.gini_impurity(y)

        # children
        left_idxs, right_idxs = self.split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate weighted average
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        gini_left = self.gini_impurity(y[left_idxs])
        gini_right = self.gini_impurity(y[right_idxs])
        child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right

        # calculate gain
        gini_gain = parent_gini - child_gini
        return gini_gain
    
    def gini_impurity(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum([p**2 for p in ps if p > 0])      # Gini = 1 - summation(p_i^2), where p_i is the probability of class i
    
    def split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])
    
    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)