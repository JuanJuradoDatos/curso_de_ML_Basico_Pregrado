###metodos de transformacion con python
##la primera parte es escalamiento y la segunda es normalizacion

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import logistic

os.chdir(r'C:\Users\Lunafernandavid\Documents\Documentos Propios Juan David\Clases_De_Python_2026\Segundo_Modulo\00_Datos')

tmp_1 = pd.read_csv('diabetes.csv')

array = tmp_1.values

X = array[:,0:8]
Y = array[:,8]

print(X)
print(Y)

###vamos a reescalar para ver las caracteristicas entre 0 y 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)
np.set_printoptions(precision=3)
print(tmp_1.columns)
print(rescaledX)

###vamos a ver el escalamiento por estandarizacion
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
print(rescaledX)

###ahora vemos la normalizacion
##aqui se ponen los valores entre cero y uno
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
print(normalizedX)


##hay otras transformaciones mas especificas, Box Cox
##la otra es la transformacion de jonson
##son usadas para corregir sesgos en la distribucion de los datos
from sklearn.preprocessing import PowerTransformer

features = tmp_1[['DiabetesPedigreeFunction', 'Age']]
pt = PowerTransformer(method='box-cox', standardize=True)
skl_boxcox = pt.fit_transform(features)
#calc_labmdas = skl_boxcox.
skl_boxcox = pt.transform(features)
df_features = pd.DataFrame(data = skl_boxcox, columns = ['DiabetesPedigreeFunction', 'Age'])
df_features

###ahora pasamos a la parte de remuestreo

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

num_fols = 10
seed = 7
kfold = KFold(n_splits=num_fols, shuffle=True, random_state=seed)
model = LogisticRegression(solver='lbfgs',max_iter=1000)
results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
print(f'Accuracy: {results.mean()}')
print(f'Sd: {results.std()}')


###ahora veamos como seria una repeticion por kfolds
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

num_fols = 10
seed = 7
num_repeats = 5
repatedkfold = RepeatedKFold(n_splits=num_fols,n_repeats=num_repeats, random_state=seed)
model = LogisticRegression(solver='lbfgs',max_iter=1000)
results = cross_val_score(model, X, Y, cv=repatedkfold, scoring='accuracy')
print(f'Accuracy: {results.mean()}')
print(f'Sd: {results.std()}')

###ahora veamos la divicion por porcentaje que es la mas usada en ML
##esta es la que se usa de facto en ML y Deeplearning

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

test_size = 0.33
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = LogisticRegression(solver='lbfgs',max_iter=1000)
model.fit(X_train, y_train)
results = model.score(X_test, y_test)
print(f'Accuracy: {results}')

###ahora una divicion aleatoria en porcentajes
##tambien llamada Shuffle Split Cross Validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

test_size = 0.33
seed = 7
n_splits = 10
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='lbfgs',max_iter=1000)
results = cross_val_score(model, X, Y, cv=kfold)
print(f'Accuracy: {results.mean()}')