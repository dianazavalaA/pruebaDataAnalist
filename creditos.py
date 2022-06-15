# Utilizamos numpy para generar datos sinteticos para saber que sucede con nuestro modelo
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time
from turtle import xcor
import numpy as np
import pandas as pd  # Utilizamos pandas para el manejo de datos
# Utilizamos matplotlib para poder gráficar nuestra información
import matplotlib.pyplot as plt
from sklearn import preprocessing  # Y utilizamos el modulo de preprocesamiento
# Importamos el modulo de KNN porque vamos a utilizar un modelo de knn para poder lograr una solución
from sklearn.neighbors import KNeighborsClassifier

# importamos el archivo con el que vamos a trabaja
# La variable se llama prospectos y esta nos ayudara a cargar el archivo con nuestra información
prospectos = pd.read_csv("datos_prestamo.csv")
print(prospectos)

# Separamos cuantos tienen un estatus de crédito positivo y negativo
print(prospectos['Estatus_prestamo'].value_counts())
# Imprimos o mostramos las columnas que tenemos
print(prospectos.columns)


x = prospectos[['Fecha_registro', 'Fecha_contacto', 'Id', 'Genero', 'Casado',
                'Dependientes', 'Educacion', 'Trabaja_para_el', 'Salario',
               'Salario_Pareja', 'Credito_pedido', 'Plazo_prestamo',
                'Historial_crediticio', 'Area_vivienda',
                'Asesor_asignado']] .values
print(x[0:5])

prospectos['Estatus_prestamo'].value_counts()

prospectos.columns

x = prospectos[['Fecha_registro', 'Fecha_contacto', 'Id', 'Genero', 'Casado',
                'Dependientes', 'Educacion', 'Trabaja_para_el', 'Salario',
               'Salario_Pareja', 'Credito_pedido', 'Plazo_prestamo',
                'Historial_crediticio', 'Area_vivienda',
                'Asesor_asignado']] .values
print(x[0:5])

y = prospectos['Estatus_prestamo'].values
print(y[0:5])

categorias = prospectos.filter(['Genero', 'Casado', 'Educacion', 'Trabaja_para_el', 'Area_vivienda',
                                'Asesor_asignado'])

print(categorias.head(10))
print(categorias.iloc[:, 0])
# Comenzamos a darle forma de puros numeros a nuestras categorias que tiene palabras
cat_numericaSex = pd.get_dummies(categorias.iloc[:, 0], drop_first=False)
print(cat_numericaSex)
cat_numericaCiv = pd.get_dummies(categorias.iloc[:, 1], prefix=[
                                 'Casado'], drop_first=False)
print(cat_numericaCiv)
cat_numericaEdu = pd.get_dummies(categorias.iloc[:, 2], drop_first=False)
print(cat_numericaEdu)
cat_numericaTra = pd.get_dummies(categorias.iloc[:, 3], prefix=[
                                 'NegocioPropio'], drop_first=False)
print(cat_numericaTra)
cat_numericaViv = pd.get_dummies(categorias.iloc[:, 4], drop_first=False)
print(cat_numericaViv)
cat_numericaAs = pd.get_dummies(categorias.iloc[:, 5], drop_first=False)
print(cat_numericaAs)

# Convertimos las fechas


def toTimeStampReg(date):
    return time.mktime(time.strptime(date, '%m/%d/%Y %H:%M %p'))


cat_numericaFR = prospectos['Fecha_registro'].apply(toTimeStampReg)

print(cat_numericaFR)

# Esta es la conversión de fechas pero de la columna fecha contacto


def toTimeStampCont(date):
    return time.mktime(time.strptime(date, '%m/%d/%Y %H:%M %p'))


cat_numericaFC = prospectos['Fecha_contacto'].apply(toTimeStampCont)
print(cat_numericaFC)

# Vamos a unir todos los valores que ya le dimos uniformidad
datosCompletos = pd.concat([cat_numericaFR, cat_numericaFC, cat_numericaSex, cat_numericaCiv,
                           cat_numericaEdu, cat_numericaTra, cat_numericaViv, cat_numericaAs], axis=1)
print(datosCompletos.head(10))

# Estos son los datos que realmente nos interesan para poder determinar si se otorga o no el crédito
datos = [[cat_numericaSex, cat_numericaCiv, cat_numericaEdu,
          cat_numericaTra, cat_numericaViv, cat_numericaAs]]
print(datos)


clase = prospectos["Estatus_prestamo"]
# print(clase)
# Comenzamos el preprocesamiento de los datos para nuestro modelo
escalador = preprocessing.MinMaxScaler()
datos = escalador.fit_transform(datosCompletos)
print(datos)

# Comenzamos a clasificar con knn
clasificador = KNeighborsClassifier(n_neighbors=5)
clasificador.fit(datos, clase)
KNeighborsClassifier()

# Vamos a separar la inf de los prestamos que fueron autorizados
prestamos_aceptados = prospectos[prospectos['Estatus_prestamo'] == 'Si']
print(prestamos_aceptados)

# Vamos a separar la inf de los prestamos que no fueron autorizados
prestamos_rechazados = prospectos[prospectos['Estatus_prestamo'] == 'No']
print(prestamos_rechazados)

# La estandarización de datos haciendo que la media sea 0 y la varianza uno es buena práctica, especialmente para algoritmos tales como KNN el cual se basa en distancia de casos
# Esta parte se vio previamente donde hice la transformación uno a uno pero en esta parte fue más rápido porque tomamos toda la información y lo convertimos a un sólo "idioma" para poder trabajarlo directamente sin separar ni concatenar
label_encoder = LabelEncoder()
prospectos_nonan = prospectos.dropna(how="all")
colums = prospectos_nonan.drop('Unnamed: 0', axis=1).columns[3:-2]
#print('columns: ', colums)
genero_encoder = LabelEncoder().fit(prospectos_nonan['Genero'])
prospectos_nonan['Genero'] = genero_encoder.transform(
    prospectos_nonan['Genero'])

casado_encoder = LabelEncoder().fit(prospectos_nonan['Casado'])
#print("es casado?", casado_encoder.transform(['Si','No']),prospectos_nonan['Casado'].unique())
prospectos_nonan['Casado'] = casado_encoder.transform(
    prospectos_nonan['Casado'])

dependientes_encoder = LabelEncoder().fit(prospectos_nonan['Dependientes'])
prospectos_nonan['Dependientes'] = dependientes_encoder.transform(
    prospectos_nonan['Dependientes'])

educacion_encoder = LabelEncoder().fit(prospectos_nonan['Educacion'])
prospectos_nonan['Educacion'] = educacion_encoder.transform(
    prospectos_nonan['Educacion'])

trabaja_encoder = LabelEncoder().fit(prospectos_nonan['Trabaja_para_el'])
prospectos_nonan['Trabaja_para_el'] = trabaja_encoder.transform(
    prospectos_nonan['Trabaja_para_el'])

historial_encoder = LabelEncoder().fit(
    prospectos_nonan['Historial_crediticio'])
prospectos_nonan['Historial_crediticio'] = historial_encoder.transform(
    prospectos_nonan['Historial_crediticio'])

area_encoder = LabelEncoder().fit(prospectos_nonan['Area_vivienda'])
prospectos_nonan['Area_vivienda'] = area_encoder.transform(
    prospectos_nonan['Area_vivienda'])

estatus_encoder = LabelEncoder().fit(prospectos_nonan['Estatus_prestamo'])
prospectos_nonan['Estatus_prestamo'] = estatus_encoder.transform(
    prospectos_nonan['Estatus_prestamo'])

asesoras_encoder = LabelEncoder().fit(prospectos_nonan['Asesor_asignado'])
prospectos_nonan['Asesor_asignado'] = asesoras_encoder.transform(
    prospectos_nonan['Asesor_asignado'])
print(prospectos_nonan.head())


prospectos_est = prospectos_nonan.dropna()
print(prospectos_est.head(), colums)
X = prospectos_est[colums].values  # .astype(float)
X = preprocessing.MinMaxScaler().fit(X).transform(X.astype(float))
print(X[1])
y = prospectos_est['Estatus_prestamo']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
print('Set de Entrenamiento:', X_train.shape,  y_train.shape)
print('Set de Prueba:', X_test.shape,  y_test.shape)

k = 4
# Entrenar el Modelo y Predecir
neigh = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
neigh

yhat = neigh.predict(X_test)
print(yhat[0:1])
print('test', X_test[:1])


# Dividimos la muestra en los conjuntos de datos para el entrenamiento y para la prueba de exactitud

print("Entrenar el set de Certeza: ", metrics.accuracy_score(
    y_train, neigh.predict(X_train)))
print("Probar el set de Certeza: ", metrics.accuracy_score(y_test, yhat))


cte = pd.array(['Hombre', 'Si', '1', 'Graduado', 'No', 4583,
               1508, 128, 360, 1, 'Rural', 'No', ''])


cte = np.array([[
    genero_encoder.transform([cte[0]]),
    casado_encoder.transform([cte[1]]),
    dependientes_encoder.transform([cte[2]]),
    educacion_encoder.transform([cte[3]]),
    trabaja_encoder.transform([cte[4]]),
    4500,
    2000,
    130,
    360,
    historial_encoder.transform([int(cte[9])]),
    area_encoder.transform([cte[10]]),
    # asesoras_encoder.transform([cte[12]])
]], dtype=object)
cte.shape = (1, 11)
print(np.array(cte,).shape)

cte_predict = neigh.predict(cte)
print(cte_predict[0:1])
print(estatus_encoder.inverse_transform(cte_predict))
