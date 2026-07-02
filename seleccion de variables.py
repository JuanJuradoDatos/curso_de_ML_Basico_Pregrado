###estas son la forma de seleccion de variables
##para poder sacar lo mejor de los datos para entrarlos a los modelos

###esta es la parte de clasificacion

import pandas as pd
import os

from markdown_it.rules_core import inline
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
from sklearn.metrics import classification_report

report = classification_report(Y_test, predictions)
print(report)



print(pd.Series(Y_test).value_counts())
tmp_1.groupby('Outcome').size()




##esta es para la parte de regresion

import pandas as pd
from sklearn.datasets import fetch_openml

tmp_1 = fetch_openml(name = 'boston', version=1, as_frame=False)

X = tmp_1.data
Y = tmp_1.target.astype(float)

##vamos a ver las metricas para regresion

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

import pandas as pd
from sklearn.datasets import fetch_openml

tmp_1 = fetch_openml(name = 'boston', version=1, as_frame=False)

X = tmp_1.data
Y = tmp_1.target.astype(float)


test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LinearRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

MAE = mean_absolute_error(Y_test, predictions)
print(MAE)

###ahora vamos a hacer KFolds
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scores = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv = kfold, scoring = scores)
print(f'Accuracy: {results.mean():.2f} ({results.std()})')

##como calcular el Error cuadratico medio y el R^2


from sklearn.datasets import fetch_openml

tmp_1 = fetch_openml(name = 'boston', version=1, as_frame=False)

X = tmp_1.data
Y = tmp_1.target.astype(float)

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LinearRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

MSE = mean_squared_error(Y_test, predictions)
print(MSE)

###ahora vamos a hacer KFolds para MSE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scores = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv = kfold, scoring = scores)
print(f'Accuracy: {results.mean():.2f} ({results.std()})')

##ahora calculamos el R^2
##para ver el ajuste del modelo en cuanto a los datos


from sklearn.datasets import fetch_openml

tmp_1 = fetch_openml(name = 'boston', version=1, as_frame=False)

X = tmp_1.data
Y = tmp_1.target.astype(float)

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LinearRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

r2 = r2_score(Y_test, predictions)
print(r2)

###ahora veamos el ajuste del modelo con cross val score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scores = 'r2'
results = cross_val_score(model, X, Y, cv = kfold, scoring = scores)
print(f'Accuracy: {results.mean():.2f} ({results.std()})')


###seleccion de variables para clasificacion y prediccion

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

##tecnicas principales fundamentadas en correlacion
##esta es para la parte de regresion y la anterior de diabetes para clasificacion

import pandas as pd
from sklearn.datasets import fetch_openml

tmp_1 = fetch_openml(name = 'boston', version=1, as_frame=False)


X = tmp_1.data
Y = tmp_1.target.astype(float)

tmp_1 = pd.DataFrame(tmp_1.data, columns=tmp_1.feature_names)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (10,10))
cor = tmp_1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

tmp_1.columns


##vamos a calcular la correlacion con ciertas variables de salida

cor_target = abs(cor['LSTAT'])
cor_target

relevant_features = cor_target[cor_target >0.60]
relevant_features


##ahora vamos a ver las caracterisitcas altamente correlacionadas entre si, esas son de eliminar
cor_matrix = tmp_1.corr().abs()
cor_matrix

upper = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k = 1).astype(np.bool))
upper

to_drop = [column for column in upper.columns if any(upper[column]>0.75)]
to_drop

tmp_1.drop(tmp_1[to_drop],axis = 1, inplace = True)
tmp_1.columns

###Analsar la eliminacion recursiva en python

##analizar bien
#sklearn.feature_selection import RFE

###vamos a ver extraccion de caracteristicas con decicion tree
##Feature importants
import pandas as pd
from sklearn.datasets import load_breast_cancer
wisconsin = load_breast_cancer()
df_wisconsin = pd.DataFrame(wisconsin.data, columns=wisconsin.feature_names)
X_wisconsin = wisconsin.data
Y_wisconsin = wisconsin.target

df_wisconsin.shape


#librerias que vamos a usar
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

##vamos a sacar features important por arboles de decision
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

##GENERAR PARTICON
test_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X_wisconsin, Y_wisconsin, test_size=test_size, random_state=seed)

##vamos a hacer que el modelo aprenda
depth = 3
tree = DecisionTreeRegressor(criterion='squared_error',max_depth = depth)
tree.fit(X_train, Y_train)

##extraer los indices de las variables utilizadas
subset = np.unique(tree.tree_.feature[tree.tree_.feature >=0])
print(f'Variables Originales: {X_wisconsin.shape[1]}')
print(f'Varibles usadas: {subset}')
print(f'Training Accuracy: {tree.score(X_train, Y_train)*100:.2f}%')
print(f'Test Accuracy: {tree.score(X_test, Y_test)*100:.2f}%')
print(df_wisconsin.columns[[ 0,1,22,23,26,27]])

###ahora extraccion de caracteristicas con Extratrees

from sklearn.ensemble import ExtraTreesClassifier
model= ExtraTreesClassifier()
model.fit(X_train, Y_train)
print(list(df_wisconsin.columns))
print(model.feature_importances_)

###ahora veamos como seria con Randonforest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

test_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X_wisconsin, Y_wisconsin, test_size=test_size, random_state=seed)

forest = RandomForestClassifier(n_estimators=100, random_state=seed)
forest.fit(X_train, Y_train)

##variables importantes
importances = forest.feature_importances_
Y_pred = forest.predict(X_test)
print(f'Error con todas las variables: {accuracy_score(Y_test, Y_pred)*100:.2f}%')

ranking = np.argsort(forest.feature_importances_)
print(wisconsin.feature_names[ranking])

plt.figure(figsize = (10,10))
plt.title('Feature Importances')
plt.barh(range(X_wisconsin.shape[1]), importances[ranking])
plt.yticks(range(X_wisconsin.shape[1]), wisconsin.feature_names[ranking], fontsize=8)

forest.fit(X_train[:, ranking[-5:]], Y_train)
Y_pred = forest.predict(X_test[:, ranking[-5:]])
print(f'Error con todas las variables: {accuracy_score(Y_test, Y_pred)*100:.2f}%')

##ahora veamos la regularizacion Lasso

from sklearn.linear_model import LassoCV

reg = LassoCV()
reg.fit(X_train, Y_train)

print(f'Best alpha using bulding Lasso CV: {reg.alphas}')
print(f'Best score using built in LassoCV: {reg.score(X_train, Y_train)*100:.2f}%')
coef = pd.Series(reg.coef_)
print(coef)

###reduccion de dimensionalidad PCA
##Estudiar esta parte de PCA para poder entender bien

