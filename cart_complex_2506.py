# Imports
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import pydotplus

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, shuffle=True)

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