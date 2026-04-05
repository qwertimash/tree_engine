import pandas as pd
from classifier_engine import impurity, get_best_split

class ClassifierTree:
    def __init__(self, max_depth=5, min_samples_split=2, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion  # 'entropy' или 'gini'
        self.tree_ = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_ser = pd.Series(y).reset_index(drop=True)
        self.tree_ = self._build_tree(X_df, y_ser, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        cur_impurity = impurity(y, self.criterion)

        if (cur_impurity == 0) or (depth >= self.max_depth) or (len(y) < self.min_samples_split):
            return {"label": y.mode().iloc[0]}
        
        best_split = get_best_split(X, y, self.criterion)

        if not best_split:
            return {"label": y.mode().iloc[0]}
        
        feature, value = best_split
        mask = X[feature] < value

        return {
            "feature": feature,
            "value": value,
            "left": self._build_tree(X[mask], y[mask], depth + 1),
            "right": self._build_tree(X[~mask], y[~mask], depth + 1)
        }
        
    def predict(self, X):
        X = pd.DataFrame(X)
        def _get_pred(row, node):
            if "label" in node:
                return node["label"]
            if float(row[node["feature"]]) < float(node["value"]):
                return _get_pred(row, node["left"])
            else:
                return _get_pred(row, node["right"])
        
        return X.apply(lambda row: _get_pred(row, self.tree_), axis=1).values