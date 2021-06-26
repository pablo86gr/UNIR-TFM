# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import pydotplus
import tensorflow as tf 

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import tree
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

seed = 128
# Cargamos el dataset
filename = "datos_complex.csv"
dataset = pd.read_csv(filename, delimiter=';')
features =[["AÑO"],["MES"],["NACIONALIDAD"],["LLEGADAS"],["MA"],["DZ"],["EH"],["MR"],["SN"],["RUTA"],["TIPO"]]
target = [["FMM"]]

# Sustituimos las variables categoricas por numeros
le=LabelEncoder()
dataset["NACIONALIDAD"]=le.fit_transform(dataset["NACIONALIDAD"])
dataset["RUTA"]=le.fit_transform(dataset["RUTA"])
dataset["TIPO"]=le.fit_transform(dataset["TIPO"])
dataset["FMM"]=le.fit_transform(dataset["FMM"])

#Separamos los datos entre categorías y clases
X = dataset[dataset.columns[:-1]]
y = dataset["FMM"]

# Dividimos el dataset entre datos de entrenamiento (80%) y de validacion(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True)



#Optimizacion de parametros
# Create hyperparameter space
epochs = [512,1024]
batches = [24, 48]
# optimizers = ['adam']
capa1 = [22, 33]
capa2 = [18, 36]
hyperparameters = dict(epochs=epochs, batch_size=batches)


for i in capa1:
    for j in capa2:
            # for m in batches:
                def create_baseline_opt():
                    #creamos el modelo
                    model = Sequential()
                    #añadimos capas
                    model.add (Dense (i, input_dim = X_train.shape [1], activation= 'relu', kernel_initializer = 'he_normal'))
                    model.add (Dense (j, input_dim = X_train.shape [1], activation= 'relu', kernel_initializer = 'he_normal'))
                    model.add (Dense (1, activation='sigmoid'))
                    # compila el modelo keras
                    model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
                    return model
                #compilamos el modelo
                estimator = KerasClassifier(build_fn=create_baseline_opt, epochs=512, batch_size=24, verbose=0)
                kfold = StratifiedKFold(n_splits=10, shuffle=True)
                #evaluamos el modelo utilizando validacion cruzada
                results = cross_val_score(estimator, X_test, y_test, cv=kfold)
                grid = GridSearchCV(estimator=estimator, cv=3, param_grid=hyperparameters)
                grid_result = grid.fit(X_train, y_train)
                print("1ª capa: %d" % (i))
                print("2ª capa: %d" % (j))
                print("Score: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
                print(grid_result.best_params_)


