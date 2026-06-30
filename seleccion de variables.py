###estas son la forma de seleccion de variables
##para poder sacar lo mejor de los datos para entrarlos a los modelos

###esta es la parte de clasificacion

import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

os.chdir(r'C:\Users\Lunafernandavid\Documents\Documentos Propios Juan David\Clases_De_Python_2026\Segundo_Modulo\00_Datos')

tmp_1 = pd.read_csv('diabetes.csv')

array = tmp_1.values

X = array[:,0:8]
Y = array[:,8]

X_clas = array[:,0:8]
Y_clas = array[:,8]

tmp_1.columns


###claramente la clase esta desvalanceada entonces debemos proceder con la metrica Kappa
tmp_1.groupby('Outcome').size()


Kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='lbfgs',max_iter=10000)
scoring = 'accuracy'
results = cross_val_score(model, X_clas, Y_clas, cv = Kfold, scoring = scoring)
print(f'Accuracy: {results.mean()*100:.2f}%')


###metrica Kappa penaliza y ve el porcentaje de acierto real

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression

test_size = 0.3
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = LogisticRegression(solver='lbfgs',max_iter=10000)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
cohen_kappa = cohen_kappa_score(Y_test, predictions)
print(f'Cohens score: {cohen_kappa*100:.2f}%')

###ahora veamos el area bajo la curva de ROC

kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='lbfgs',max_iter=10000)
scoring = 'roc_auc'
results = cross_val_score(model, X_test, Y_test, cv = kfold, scoring = scoring)
print(f'Accuracy: {results.mean()} {results.std()}')


###ahora vamos a ver la matriz de confusion

from sklearn.metrics import confusion_matrix

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = LogisticRegression(solver='lbfgs',max_iter=10000)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
matriz = confusion_matrix(Y_test, predictions)
print(matriz)

##ahora veamos el reporte del modelo de clasificacion



print(pd.Series(Y_test).value_counts())
tmp_1.groupby('Outcome').size()




##esta es para la parte de regresion

import pandas as pd
from sklearn.datasets import fetch_openml

tmp_1 = fetch_openml(name = 'boston', version=1, as_frame=False)

X = tmp_1.data
Y = tmp_1.target.astype(float)

