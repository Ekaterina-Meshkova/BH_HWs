# banana_set.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score


#Загрузить данные из csv
def load_csv(file_path):
        """Загрузка данных из CSV файла."""
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"Ошибка при загрузке CSV: {e}")
            return None

# Визуализация1 датасета (столбчатые диаграммы)
def viev(data):

# установить стиль Seaborn
  sns.set(style="whitegrid")

# df — это DataFrame
# создать ящики с усами для каждой колонки df
  plt.figure(figsize=(12, 10))

# создать ящик с усами для каждого чмслового столбца
  for index, column in enumerate(data.select_dtypes(include=[np.number]).columns):
      plt.subplot((len(data.columns) // 3) + 1, 3, index + 1)
      sns.boxplot(y=data[column])

  plt.tight_layout()
  plt.show()

# Визуализация2 датассета (гистограммы)
def viev2(data1):
  # установить стиль Seaborn
  sns.set(style="whitegrid")

  # создать гистограмму для каждой числовой переменной
  data1.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')

  # добавить названия для каждого графика и осей
  for ax in plt.gcf().get_axes():
      ax.set_xlabel('Значение')
      ax.set_ylabel('Частота')
  # отрегулировать макета для предотвращения наложения подписей
  plt.tight_layout()
  # показать график
  plt.show()

# Подготовка данных (удаление айдишнков и строк с пустыми значениями)
def clean_data(data):
    data = data.drop(columns=['A_id'])
    data = data.dropna()
    data = data.astype({'Acidity'})
    def label(Quality):
        if Quality == "Good":
            return 0
        if Quality == "Dad":
            return 1
        return None

    data['Label'] = data['Quality'].apply(label)

    data = data.drop(columns=['Quality'])

    data = data.astype({'Label': 'int64'})

    return data

def learning(data1):
  # Разделение данных на признаки и целевую переменную
  X = data1.iloc[:, :-1]
  y = data1['Label']

  # Разделение на обучающую и тестовую выборки
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

  # Стандартизация данных
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # Создание и обучение моделей
  log_reg = LogisticRegression()
  log_reg.fit(X_train_scaled, y_train)

  rf = RandomForestClassifier()
  rf.fit(X_train_scaled, y_train)

  # Предсказание на тестовых данных
  y_pred_log_reg = log_reg.predict(X_test_scaled)
  y_pred_rf = rf.predict(X_test_scaled)

  # Оценка моделей
  accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
  accuracy_rf = accuracy_score(y_test, y_pred_rf)
  report_log_reg = classification_report(y_test, y_pred_log_reg)
  report_rf = classification_report(y_test, y_pred_rf)

  print(accuracy_log_reg, accuracy_rf, report_log_reg, report_rf)
