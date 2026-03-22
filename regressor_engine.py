import pandas as pd
import numpy as np

def get_tresholds(df,target='target'):
    """
    Разбиение на уникальные значение признаков (средние значения между уникальными)
    """
    cols = {}
    for col in df.columns: 
        if col == target: continue
        values = sorted(df[col].unique())
        mids = [(values[i] + values[i+1])/2 for i in range(len(values) - 1)]
        cols[col] = mids
    return cols

def mse(df, target):
    """
    Подсчёт mse - функции ошибки для регрессии в df по target
    """
    if len(df) == 0: return 0
    y = df[target]
    return ((y - y.mean())**2).mean()

def variance_reduction(left, right, current_mse, target):
    """
    Подсчёт прироста информации
    """
    p = len(left) / (len(left) + len(right))
    return current_mse - p * mse(left, target) - (1 - p) * mse(right, target)

def get_best_split_regressor(df, cols, target):
    """
    Нахождение лучшего разбиения
    """
    best_reduction = -1
    best_question = None
    current_mse = mse(df, target)
    
    for feature in cols:
        for value in cols[feature]:
            left = df[df[feature] < value]
            right = df[~(df[feature] < value)]

            if len(left) == 0 or len(right) == 0:
                continue

            reduction = variance_reduction(left, right, current_mse, target)

            if reduction > best_reduction:
                best_reduction = reduction
                best_question = (feature, value)
    return best_question
