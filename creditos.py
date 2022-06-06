# Utilizamos numpy para generar datos sinteticos para saber que sucede con nuestro modelo
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

y = prospectos['Estatus_prestamo'].values
print(y[0:5])

print(pd.get_dummies(prospectos, columns=['Genero']))  # Esta es para genero

categorias = prospectos.filter(['Fecha_registro', 'Fecha_contacto', 'Genero', 'Casado', 'Educacion', 'Trabaja_para_el', 'Area_vivienda',
                                'Asesor_asignado'])

cat_numerica = pd.get_dummies(categorias, drop_first=True)
print(cat_numerica)
