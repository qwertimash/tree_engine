import pandas as pd
import math

def get_tresholds(X):
    """
    Разбиение на уникальные значение признаков (средние значения между уникальными)
    """
    cols = {}
    for col in X.columns: 
        values = sorted(X[col].unique())
        mids = [(values[i] + values[i+1])/2 for i in range(len(values) - 1)]
        cols[col] = mids
    return cols

def entropy(y):
    """
    Подсчёт энтропии - функции ошибки для классификации по y
    """
    counts = pd.Series(y).value_counts(normalize=True)
    ent = 0
    for p in counts:
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def gini(y):
    """
    Подсчёт критерия Джини: 1 - sum(p_i^2)
    Чем чище узел - тем ближе к 0
    """
    counts = pd.Series(y).value_counts(normalize=True)
    return 1 - sum(p ** 2 for p in counts)

def impurity(y, criterion='entropy'):
    """
    Выбор функции нечистоты по критерию
    """
    if criterion == 'gini':
        return gini(y)
    return entropy(y)

def info_gain(y, y_left, y_right, criterion='entropy'):
    """
    Подсчёт прироста информации
    """
    if len(y) == 0: return 0
    p = len(y_left) / len(y)
    return impurity(y, criterion) - p * impurity(y_left, criterion) - (1 - p) * impurity(y_right, criterion)

def get_best_split(X, y, criterion='entropy'):
    """
    Нахождение лучшего разбиения
    """
    cols = get_tresholds(X)
    best_gain = 0
    best_question = None
    for feature in cols:
        for value in cols[feature]:
            y_left = y[X[feature] < value]
            y_right = y[~(X[feature] < value)]

            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            gain = info_gain(y, y_left, y_right, criterion)

            if gain >= best_gain:
                best_gain = gain
                best_question = (feature, value)
    if best_gain == 0: return None
    return best_question