###Primero cargar el conjunto de datos
##en nuestro caso es el de diavetes un set muy utilizado como data set toy
from tkinter.font import names

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


os.chdir(r'C:\Users\Lunafernandavid\Documents\Documentos Propios Juan David\Clases_De_Python_2026\Segundo_Modulo\00_Datos')

tmp_1 = pd.read_csv('diabetes.csv')

###ahora hagamso un breve analisis descritivo

##para ver los primeros registros del dataframe
tmp_1.head()

##para ver las dimensiones del dataframe
tmp_1.shape

##que tipo de dato es cada una de las variables
tmp_1.dtypes

##ahora hacemos una corta descripcion de los datos de forma
##Estadistica para hacernos una idea del dataframe
tmp_1.describe()

##vamos a ver como esta distribuido la clase objetivo
##la que se busca predecir
tmp_1.groupby('Outcome').size()

##ahora vamos a ver la correlacion entre las variables
cor_1 = tmp_1.corr(method='pearson')
cor_1

###ahora vamos a ver si las variables tienen sesgo
tmp_1.skew()

###veamos la visualizacion de las variables
##primero un diagrama de barras
##primero visualizacion univariable para ver el comportamiento de cada variable
##de forma univariada


fig = plt.figure(figsize = (10,10))
ax = fig.gca()
tmp_1.hist(ax = ax)

####ahora vamos a ver las graficas pero con seaborn
f, axes = plt.subplots(3,3,figsize = (14,14))
sns.distplot(tmp_1['Pregnancies'],ax = axes[0,0])
sns.distplot(tmp_1['Glucose'],ax = axes[0,1])
sns.distplot(tmp_1['BloodPressure'],ax = axes[0,2])
sns.distplot(tmp_1['SkinThickness'],ax = axes[1,0])
sns.distplot(tmp_1['Insulin'],ax = axes[1,1])
sns.distplot(tmp_1['BMI'],ax = axes[1,2])
sns.distplot(tmp_1['DiabetesPedigreeFunction'],ax = axes[2,0])
sns.distplot(tmp_1['Age'],ax = axes[2,1])
sns.distplot(tmp_1['Outcome'],ax = axes[2,2])

###vamos a ver el diagrama de densidad y el boxplot

fig = plt.figure(figsize = (16,16))
ax = fig.gca()
tmp_1.plot(ax = ax, kind = 'density', subplots = True, layout = (3,3), sharex = False)

###vamos a ver los box-plot
fig = plt.figure(figsize = (16,16))
ax = fig.gca()
tmp_1.plot(ax = ax, kind = 'box', subplots = True, layout = (3,3), sharex = False)

####vamos a ver la visualizacion de los datos multivariable
##vamos a ver la matriz de correlacion con mapa de calor

corr_2 = tmp_1.corr(method='pearson')
fig = plt.figure()
ax = fig.gca()
cax = ax.matshow(corr_2,vmin = -1, vmax = 1)
fig.colorbar(cax)
ticks = np.arange(0,len(tmp_1.columns),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(tmp_1.columns)
ax.set_yticklabels(tmp_1.columns)
plt.show()

###ahora veamos ese mapa de calor con seaborn
corr_2 = tmp_1.corr(method='pearson')
plt.figure(figsize=(10,10))
ax = sns.heatmap(corr_2,vmin = -1, vmax = 1,square = True,annot=True, cmap = 'RdYlGn')
plt.title('Correlation Matrix')
plt.show()

##ahora veamos la matriz de dispercion y tendencias lineales

from pandas.plotting import scatter_matrix

plt.rcParams['figure.figsize'] = [20,15]
scatter_matrix(tmp_1)
plt.show()

##ahora veamos la dispercion y tendencias con seaborn

sns.pairplot(tmp_1)

###vamos a ver la matriz de dispercion por clase y el boxplot por clase
##es decir cuando veamos el grafico de dispercion ver cuales pertenecen a una u otra clase
sns.pairplot(tmp_1, hue = 'Outcome', diag_kind = 'hist')





tmp_1

tmp_1.head()

tmp_1.describe()

tmp_1.shape

tmp_1.columns

os.listdir()