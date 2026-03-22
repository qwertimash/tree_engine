import pandas as pd
import math

def get_tresholds(df,target='target'):
    """
    Разбиение на уникальные значение признаков (средние значения между уникальными)
    """
    cols = {}
    for col in df.columns: 
        if col == target:continue
        values = sorted(df[col].unique())
        mids = [(values[i] + values[i+1])/2 for i in range(len(values) - 1)]
        cols[col] = mids
    return cols

def entropy(df,target):
    """
    Подсчёт энтропии - функции ошибки для классификации в df по target
    """
    counts = df[target].value_counts(normalize=True)
    ent = 0
    for p in counts:
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def info_gain(left,right,current_entropy,target):
    """
    Подсчёт прироста информации
    """
    p = len(left)/(len(left)+len(right))
    return current_entropy - p*entropy(left,target) - (1-p) * entropy(right,target)

def get_best_split(df,target):
    """
    Нахождение лучшего разбиения
    """
    cols = get_tresholds(df,target)
    best_gain = 0
    best_question = None
    current_entropy = entropy(df,target)
    for feature in cols:
        for value in cols[feature]:
            left = df[df[feature] < value]
            right = df[~(df[feature] < value)]

            if len(left) == 0 or len(right) == 0:
                continue
            
            gain = info_gain(left,right,current_entropy,target)

            if gain > best_gain:
                best_gain = gain
                best_question = (feature,value)
    return best_question