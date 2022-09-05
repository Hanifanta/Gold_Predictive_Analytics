# -*- coding: utf-8 -*-
"""Predictive_Analytics_Hanif

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p5AADScffh37JJe73x5wP1mJ3xKC50ZH
"""

!kaggle datasets download -d mattiuzc/commodity-futures-price-history

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
# %matplotlib inline

import zipfile

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

local_zip = '/content/commodity-futures-price-history.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

df = pd.read_csv('/content/Commodity Data/Gold.csv')
df

print(f'The data has {df.shape[0]} records and {df.shape[1]} columns.')

df.info()

"""# **Exploratory Data Analysis**

**Deskripsi Variabel**

* Date : Tanggal pencatatan Data
* Open : Harga buka dihitung perhari
* High : Harga tertinggi perhari
* Low : Harga terendah perhari
* Close : Harga tutup dihitung perhari
* Adj Close : Harga penutupan pada hari tersebut setelah disesuaikan dengan aksi korporasi seperti right issue, stock split atau stock reverse.
* Volume : Volume transaksi

**Pengecekan missing data**
"""

df.isnull().sum()

col_missing = [col for col in df.columns if df[col].isnull().any()]

imputer = SimpleImputer()
df[col_missing] = imputer.fit_transform(df[col_missing])
df.head()

df.isnull().sum()

"""# **Explore Statistic Information**

**masing-masing kolom memiliki informasi, antara lain:**

* **count** adalah jumlah sampel pada data.
* **mean** adalah nilai rata-rata.
* **std** adalah standar deviasi.
* **min** adalah nilai minimum.
* **25%** adalah kuartil pertama.
* **50%** adalah kuartil kedua (nilai tengah).
* **75%** adalah kuartil ketiga.
* **max** adalah nilai maksimum
"""

df.describe()

"""# **Data visualiation**"""

numerical_col = [col for col in df.columns if df[col].dtypes == 'float64']
plt.subplots(figsize=(10,7))
sns.boxplot(data=df[numerical_col]).set_title("Gold Price")
plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3-Q1
df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]

df.shape

numerical_col = [col for col in df.columns if df[col].dtypes == 'float64']
plt.subplots(figsize=(10,7))
sns.boxplot(data=df[numerical_col]).set_title("Gold Price")
plt.show()

"""# **Univariate Analysis**

Fitur yang diprediksi pada kasus ini adalah terfokus pada 'Adj Close'
"""

cols = 3
rows = 2
fig = plt.figure(figsize=(cols * 5, rows * 5))

for i, col in enumerate(numerical_col):
  ax = fig.add_subplot(rows, cols, i + 1)
  sns.histplot(x=df[col], bins=30, kde=True, ax=ax)
fig.tight_layout()
plt.show()

"""# **Multivariate Analysis**

Selanjutnya kita akan menganalisis korelasi fitur "Adj Close" terhadap fitur lain seperti "Open", "High", "Low", "Close" dan "Volume". Dapat disimpulkan bahwa "Adj Close" memiliki korelasi positif yang kuat terhadap "Open", "High", "Low" dan "Close", sedangkan untuk fitur "Volume" memiliki korelasi sedang terhadap fitur "Adj Close"


"""

sns.pairplot(df[numerical_col], diag_kind='kde')
plt.show()

plt.figure(figsize=(15,8))
corr = df[numerical_col].corr().round(2)
sns.heatmap(data=corr, annot=True, vmin=-1, vmax=1, cmap='coolwarm', linewidth=1)
plt.title('Correlation matrix for numerical feature', size=15)
plt.show()

df = df.drop(['Date','Volume', 'Close'], axis=1)
df.head()

"""# **Splitting Dataset**"""

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

print(len(X_train), 'records')
print(len(y_train), 'records')
print(len(X_test), 'records')
print(len(y_test), 'records')

"""# **Data Normalization**

Untuk melakukan normalisasi data kita akan menggunakan library MinMaxScaler. Fungsi normalisasi pada data agar model lebih cepat dalam mempelajari data karena data telah diubah pada rentang tertentu seperti antara 0 dan 1
"""

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = pd.DataFrame(columns=['train_mse', 'test_mse'], index=['SVR', 'KNN', 'GradientBoosting'])

"""# **Modeling**
**Hyperparameter Tuning**
Hyperparameter tuning adalah salah satu teknik yang dilakukan akan model dapat berjalan dengan performa terbaik. Biasanya dalam hyperparameter tuning, hyperparameter akan ditentukan secara acak oleh teknisi. Namun jika tidak ingin mencoba coba hyperparameter mana yang terbaik, kita dapat menggunakan GridSearch. GridSearch merupakan sebuah teknik yang memungkinkan kita untuk menguji beberapa hyperparameter sekaligus pada sebuah model
"""

def grid_search(model, hyperparameters):
  results = GridSearchCV(
      model,
      hyperparameters,
      cv=5,
      verbose=1,
      n_jobs=6
  )

  return results

svr = SVR()
hyperparameters = {
    'kernel': ['rbf'],
    'C': [0.001, 0.01, 0.1, 10, 100, 1000],
    'gamma': [0.3, 0.03, 0.003, 0.0003]
}

svr_search = grid_search(svr, hyperparameters)
svr_search.fit(X_train, y_train)
print(svr_search.best_params_)
print(svr_search.best_score_)

gradient_boost = GradientBoostingRegressor()
hyperparameters = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'n_estimators': [250, 500, 750, 1000],
    'criterion': ['friedman_mse', 'squared_error']
}

gradient_boost_search = grid_search(gradient_boost, hyperparameters)
gradient_boost_search.fit(X_train, y_train)
print(gradient_boost_search.best_params_)
print(gradient_boost_search.best_score_)

knn = KNeighborsRegressor()
hyperparameters = {
    'n_neighbors': range(1, 10)
}

knn_search = grid_search(knn, hyperparameters)
knn_search.fit(X_train, y_train)
print(knn_search.best_params_)
print(knn_search.best_score_)

"""# **Model Training**"""

svr = SVR(C=10, gamma=0.3, kernel='rbf')
svr.fit(X_train, y_train)

gradient_boost = GradientBoostingRegressor(criterion='squared_error', learning_rate=0.01, n_estimators=1000)
gradient_boost.fit(X_train, y_train)

knn = KNeighborsRegressor(n_neighbors=9)
knn.fit(X_train, y_train)

"""# **Model Evaluation**"""

model_dict = {
    'SVR': svr,
    'GradientBoosting': gradient_boost,
    'KNN': knn,
}

for name, model in model_dict.items():
  models.loc[name, 'train_mse'] = mean_squared_error(y_train, model.predict(X_train))
  models.loc[name, 'test_mse'] = mean_squared_error(y_test, model.predict(X_test))

models.head()

models.sort_values(by='test_mse', ascending=False).plot(kind='bar', zorder=3)