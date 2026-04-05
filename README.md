# tree_engine 🌳

Реализация Decision Tree с нуля на Python — без использования sklearn для логики дерева.

## Результаты

Сравнение с `sklearn.tree.DecisionTreeClassifier` на датасете Titanic (критерий: entropy, F1-score):

| Модель | F1-score |
|--------|----------|
| **tree_engine** | **0.7273** |
| sklearn | 0.7183 |

## Что реализовано

**Классификатор** (`MyTree.py`, `classifier_engine.py`)
- Критерий разбиения: энтропия + information gain
- Параметры: `max_depth`, `min_samples_split`
- Предсказание через рекурсивный обход дерева

**Регрессор** (`regressor_engine.py`)
- Критерий разбиения: MSE + variance reduction
- Логика аналогична классификатору

## Как запустить

```bash
pip install pandas numpy scikit-learn
```

```python
from MyTree import ClassifierTree

model = ClassifierTree(max_depth=5, min_samples_split=2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Структура

```
tree_engine/
├── MyTree.py              # ClassifierTree — основной класс
├── classifier_engine.py   # энтропия, information gain, поиск лучшего сплита
├── regressor_engine.py    # MSE, variance reduction для регрессии
└── tree_classifier.ipynb  # сравнение с sklearn на датасете Titanic
```

## Детали реализации

Пороги для сплитов вычисляются как средние значения между соседними уникальными значениями признака — стандартный подход как в sklearn. Лучший сплит выбирается полным перебором всех признаков и порогов.

---

*Проект сделан в рамках изучения ML с нуля — понять как работает дерево изнутри, а не просто вызвать `fit()`.*