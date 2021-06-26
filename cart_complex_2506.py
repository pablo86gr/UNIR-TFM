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
# Información sobre el dataset
print("Resumen de información del dataset:")
dataset.info()
print(dataset.head(10))
print("-----------------------------------------")
print()

# Sustituimos las variables categoricas por numeros
le=LabelEncoder()
dataset["NACIONALIDAD"]=le.fit_transform(dataset["NACIONALIDAD"])
dataset["RUTA"]=le.fit_transform(dataset["RUTA"])
dataset["TIPO"]=le.fit_transform(dataset["TIPO"])
dataset["FMM"]=le.fit_transform(dataset["FMM"])

print("\n\nResumen de información del dataset tras la transformacion:")
print(dataset.info())

#Separamos los datos entre categorías y clases
X = dataset[dataset.columns[:-1]]
y = dataset["FMM"]

# Dividimos el dataset entre datos de entrenamiento (80%) y de validacion(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True)

# ÁRBOLES DE DECISION - CART
print("\n\nArboles de decisión. Modelo CART:")
print()
models = []
classifier = DecisionTreeClassifier()
models.append(("CART", classifier))
# Entrenamos el modelo
classifier.fit(X_train, y_train)

# Calculamos el resultado que predeciría el modelo con los datos de entrenamiento
y_pred = classifier.predict(X_test)

# Calculamos las metricas para evaluar cómo de bueno es el modelo
print("\n\nMatriz de confusión:\n",confusion_matrix(y_test, y_pred))
print()
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['True', 'False']))

# Guardamos una imagen del árbol
plt.figure()
data = tree.export_graphviz(classifier, out_file=None, feature_names=dataset.columns.values[:-1], class_names=["True", "False"], filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(data)
plt.title("CART Decision Tree")
graph.write_png("CARTdecisiontree.png", )
img = pltimg.imread("CARTdecisiontree.png")
imgplot = plt.imshow(img)
print("-----------------------------------------")
print()


#---------------
#- RED NEURONAL
#---------------
# model = Sequential ()
# model.add (Dense (6, input_dim = X_train.shape [1], activation= 'relu', kernel_initializer = 'he_normal'))
# model.add (Dense (36, input_dim = X_train.shape [1], activation= 'relu', kernel_initializer = 'he_normal'))
# model.add (Dense (1, activation='sigmoid'))
# # compila el modelo keras
# model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# # se ajusta al modelo de Keras en el conjunto de datos
# model.fit (X_train, y_train, epochs = 50, batch_size = 12, verbose = 2)
# # evaluar el modelo de keras
# _, precision = model.evaluate (X_test, y_test, verbose = 0)
# print ('Precisión de la red neuronal:% .2f'% (precision * 100))

def create_baseline():
    #creamos el modelo
    model = Sequential()
    #añadimos capas
    model.add (Dense (11, input_dim = X_train.shape [1], activation= 'relu', kernel_initializer = 'he_normal'))
    model.add (Dense (36, input_dim = X_train.shape [1], activation= 'relu', kernel_initializer = 'he_normal'))
    model.add (Dense (1, activation='sigmoid'))
    # compila el modelo keras
    model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

#compilamos el modelo
estimator = KerasClassifier(build_fn=create_baseline, epochs=256, batch_size=24, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
#evaluamos el modelo utilizando validacion cruzada
results = cross_val_score(estimator, X_test, y_test, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def create_baseline_opt():
    #creamos el modelo
    model = Sequential()
    #añadimos capas
    model.add (Dense (capa1, input_dim = X_train.shape [1], activation= 'relu', kernel_initializer = 'he_normal'))
    model.add (Dense (36, input_dim = X_train.shape [1], activation= 'relu', kernel_initializer = 'he_normal'))
    model.add (Dense (1, activation='sigmoid'))
    # compila el modelo keras
    model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

#Optimizacion de parametros
# Create hyperparameter space
epochs = [256,512]
batches = [12,24]
optimizers = [['rmsprop'], ['adam']]
capa1 = [11, 22, 33]
capa2 = [18, 36, 72]

# # Create hyperparameter options
# # hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)
# hyperparameters = dict(epochs=epochs, batch_size=batches)
# # Create grid search
# grid = GridSearchCV(estimator=estimator, cv=3, param_grid=hyperparameters)
# # Fit grid search
# grid_result = grid.fit(X_train, y_train)
# print(grid_result.best_params_)

for x in capa1:
    #compilamos el modelo
    estimator = KerasClassifier(build_fn=create_baseline_opt, epochs=256, batch_size=24, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    #evaluamos el modelo utilizando validacion cruzada
    results = cross_val_score(estimator, X_test, y_test, cv=kfold)
    print("We're on time %d" % (x))
    print("Variando las neuronas de la capa 1: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    