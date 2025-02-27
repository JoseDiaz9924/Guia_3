# Databricks notebook source
# MAGIC %md
# MAGIC # **Guía 3**
# MAGIC
# MAGIC ## **¿Cómo podemos controlar el creciente número de accidentes en Nueva York?**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introduccion

# COMMAND ----------

# MAGIC %md
# MAGIC **Contexto empresarial.** La ciudad de Nueva York ha experimentado un aumento en el número de accidentes en las carreteras de la ciudad. Quieren saber si el número de accidentes ha aumentado en las últimas semanas. Para todos los accidentes reportados, han recopilado detalles para cada accidente y han estado manteniendo registros durante el último año y medio (desde enero de 2018 hasta agosto de 2019).
# MAGIC
# MAGIC La ciudad te ha contratado para que construyas visualizaciones que les ayuden a identificar patrones en accidentes, lo que les ayudaría a tomar acciones preventivas para reducir la cantidad de accidentes en el futuro. Tienen ciertos parámetros como municipio, hora del día, motivo del accidente, etc. De los que se preocupan y de los que les gustaría obtener información específica.

# COMMAND ----------

# MAGIC %md
# MAGIC **Problema comercial.** Su tarea es formatear los datos proporcionados y proporcionar visualizaciones que respondan las preguntas específicas que tiene el cliente, que se mencionan a continuación.

# COMMAND ----------

# MAGIC %md
# MAGIC **Contexto analítico.** Se le proporciona un archivo CSV (accidente) que contiene detalles sobre cada accidente, como fecha, hora, ubicación del accidente, motivo del accidente, tipos de vehículos involucrados, recuento de lesiones y muertes, etc. El delimitador en el archivo CSV dado es `;` en lugar del predeterminado **`,`**.
# MAGIC
# MAGIC Realizará las siguientes tareas con los datos:
# MAGIC
# MAGIC 1. Leer, transformar y preparar datos para su visualización
# MAGIC 2. Realizar análisis y construir visualizaciones de los datos para identificar patrones en el conjunto de datos.
# MAGIC         
# MAGIC El cliente tiene un conjunto específico de preguntas a las que le gustaría obtener respuestas. Deberá proporcionar visualizaciones para acompañar estos:
# MAGIC
# MAGIC 1. ¿Cómo ha fluctuado el número de accidentes durante el último año y medio? ¿Han aumentado con el tiempo?
# MAGIC 2. Para un día en particular, ¿durante qué horas es más probable que ocurran accidentes?
# MAGIC 3. ¿Hay más accidentes entre semana que durante los fines de semana?
# MAGIC 4. ¿Cuál es la proporción de recuento de accidentes por área por municipio? ¿Qué distritos tienen un número desproporcionadamente grande de accidentes para su tamaño?
# MAGIC 5. Para cada municipio, ¿durante qué horas es más probable que ocurran accidentes?
# MAGIC 6. ¿Cuáles son las 5 principales causas de accidentes en la ciudad?
# MAGIC 7. ¿Qué tipos de vehículos están más involucrados en accidentes por municipio?
# MAGIC 8. ¿Qué tipos de vehículos están más involucrados en las muertes?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview de la data

# COMMAND ----------

# MAGIC %md
# MAGIC Analizemos las columnas presentes en el data frame

# COMMAND ----------

# MAGIC %md
# MAGIC Este conjunto de datos contiene información detallada sobre accidentes de tránsito registrados en la ciudad de Nueva York. A continuación, se presenta la descripción de cada columna:
# MAGIC
# MAGIC - **BOROUGH**. Municipio donde ocurrió el accidente (ejemplo: Manhattan, Brooklyn, Queens, Bronx, Staten Island).
# MAGIC - **COLLISION_ID** Identificador único asignado a cada colisión para diferenciar los accidentes registrados.
# MAGIC - **CONTRIBUTING FACTOR VEHICLE** (1, 2, 3, 4, 5) Factores que contribuyeron a la ocurrencia del accidente, como exceso de velocidad, distracción del conductor, fallas mecánicas, malas condiciones climáticas, entre otros.
# MAGIC Se pueden registrar hasta cinco factores por accidente, cada uno correspondiente a un vehículo involucrado.
# MAGIC - **CROSS STREET NAME**  Nombre de la calle transversal más cercana al lugar del accidente, útil para ubicar intersecciones peligrosas.
# MAGIC - **DATE** Fecha exacta en la que ocurrió el accidente en formato YYYY-MM-DD.
# MAGIC - **TIME** Hora del accidente en formato HH:MM AM/PM, permitiendo analizar patrones horarios en la siniestralidad.
# MAGIC - **LATITUDE y LONGITUDE**
# MAGIC
# MAGIC - **NUMBER OF (CYCLISTS, MOTORISTS, PEDESTRIANS) INJURED** Número de personas heridas en el accidente, clasificadas en tres categorías: Ciclistas, Motociclistas - conductores de vehículos y Peatones.
# MAGIC
# MAGIC - **NUMBER OF (CYCLISTS, MOTORISTS, PEDESTRIANS) DEATHS** Número de víctimas fatales en el accidente, categorizadas en: ciclistas, Motociclistas - conductores de vehículos y Peatones.
# MAGIC
# MAGIC - **ON STREET NAME**  Nombre de la calle donde ocurrió el accidente, información clave para el análisis de zonas de alto riesgo.
# MAGIC
# MAGIC - **VEHICLE TYPE CODE (1, 2, 3, 4, 5)** Tipos de vehículos involucrados en el accidente, pudiendo haber hasta cinco vehículos registrados por accidente.
# MAGIC Sedán, SUV, Camión, Motocicleta, Autobús, Bicicleta, etc.
# MAGIC
# MAGIC - **ZIP CODE**  Código postal correspondiente a la ubicación del accidente, útil para agrupar eventos por áreas específicas dentro de la ciudad.

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Limpieza del dataset**

# COMMAND ----------

# MAGIC %md
# MAGIC Para asegurar que los datos sean consistentes y puedan ser analizados correctamente, realizaremos la imputación de los valores faltantes en las columnas que presentan datos nulos. A continuación, se detallan los pasos que debes seguir para limpiar el conjunto de datos.
# MAGIC - **Paso 1: Identificar los valores faltantes**
# MAGIC - **Paso 2: Decidir el método de imputación**
# MAGIC
# MAGIC Dado el análisis de valores nulos, se aplicarán diferentes estrategias de imputación según el tipo de dato. Por ejemplo: para la columna ZIP CODE, se imputará con el código postal más frecuente (moda) dentro de cada municipio registrado en BOROUGH. En el caso de las coordenadas LATITUDE y LONGITUDE, se reemplazarán los valores faltantes con la media de las coordenadas dentro de cada municipio. La columna ON STREET NAME será rellenada con "UNKNOWN" en caso de estar vacía. Para los factores que contribuyeron al accidente (CONTRIBUTING FACTOR VEHICLE X), los valores nulos serán sustituidos por "Unspecified". Finalmente, en las columnas de VEHICLE TYPE CODE X, los valores ausentes se reemplazarán con "Unknown" para asegurar la integridad del análisis.

# COMMAND ----------

# Solución propuesta
import pandas as pd
# Lectura del Archivo con Pandas
file_path = "/Workspace/Users/omaragonm@compensar.com/accidents-1.csv"
df = pd.read_csv(file_path, sep=';')

# Identificar los valores faltantes
missing_values = df.isnull().sum()
print(missing_values)

df['ZIP CODE'] = df.groupby('BOROUGH')['ZIP CODE'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

df['LATITUDE'] = df.groupby('BOROUGH')['LATITUDE'].transform(lambda x: x.fillna(x.mean()))
df['LONGITUDE'] = df.groupby('BOROUGH')['LONGITUDE'].transform(lambda x: x.fillna(x.mean()))

df['ON STREET NAME'] = df['ON STREET NAME'].fillna('UNKNOWN')

contributing_factors = [col for col in df.columns if 'CONTRIBUTING FACTOR VEHICLE' in col]
df[contributing_factors] = df[contributing_factors].fillna('Unspecified')

vehicle_types = [col for col in df.columns if 'VEHICLE TYPE CODE' in col]
df[vehicle_types] = df[vehicle_types].fillna('Unknown')

# Verificar los valores faltantes después de la imputación
missing_values_after_imputation = df.isnull().sum()
print("\nValores faltantes después de imputar:")
print(missing_values_after_imputation)


# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 1
# MAGIC
# MAGIC Desde 2014, la ciudad de Nueva York ha estado implementando el programa de seguridad vial Vision Zero, cuyo objetivo es reducir a cero el número de muertes por accidentes de tránsito para el año 2024. Como parte de esta iniciativa, se han implementado y mejorado diversas estrategias para aumentar la seguridad en las calles.
# MAGIC
# MAGIC A continuación, se presentan algunas de las medidas adoptadas en el plan:
# MAGIC
# MAGIC - [ X] Detección automatizada de peatones para mejorar la seguridad en los cruces. 
# MAGIC **Usando NUMBER OF PEDESTRIANS INJURED o NUMBER OF PEDESTRIANS KILLED**
# MAGIC - [X ] Auditorías de seguridad vial en zonas con alta incidencia de accidentes.
# MAGIC **Usando COLLISION_ID **
# MAGIC - [X ] Expansión de la red de carriles para bicicletas para reducir la exposición de ciclistas a incidentes con vehículos.
# MAGIC **Usando NUMBER OF CYCLIST INJURED y NUMBER OF CYCLIST KILLED**
# MAGIC - [ X] Programas de educación y sensibilización para fomentar el respeto a las normas de tránsito.
# MAGIC **Usando CONTRIBUTING FACTOR VEHICLE 1**
# MAGIC - [X ] Construcción de islas de refugio peatonal para mejorar la seguridad en calles de alto tráfico.
# MAGIC **Usando NUMBER OF PEDESTRIANS INJURED**
# MAGIC - [X ] Implementación de reductores de velocidad inteligentes, como topes y amortiguadores, basados en el análisis de datos.
# MAGIC **Usando CONTRIBUTING FACTOR VEHICLE 1, CONTRIBUTING FACTOR VEHICLE 2**
# MAGIC
# MAGIC **Pregunta: ¿Cuáles de estas iniciativas podrían beneficiarse directamente del análisis de los datos disponibles sobre accidentes? Marque todas las opciones que considere aplicables.**
# MAGIC
# MAGIC Instrucciones: Para marcar una opción, agregue una "[x]" en la casilla correspondiente.

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Sigamos adelante y respondamos a cada una de las preguntas del cliente.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 2:
# MAGIC
# MAGIC Agrupe los datos disponibles mensualmente y genere un line plot de accidentes a lo largo del tiempo. ¿Ha aumentado el número de accidentes durante el último año y medio?
# MAGIC
# MAGIC **Sugerencia**: Puede encontrar útiles las funciones de pandas ```to_datetime ()``` y ```dt.to_period ()```.
# MAGIC

# COMMAND ----------

import matplotlib.pyplot as plt

# Solución propuesta
# Convertimos a tipo fecha
df['DATE'] = pd.to_datetime(df['DATE'])

#Extraemos Mes y año
df['MONTH'] = df['DATE'].dt.to_period('M')

#Nro de accidentes x mes
accidentes_x_mes = df.groupby('MONTH').size()

# Generar el gráfico de línea
plt.figure(figsize=(10, 6))
accidentes_x_mes.plot(kind='line', marker='o')

# Etiquetas y título
plt.title('Número de Accidentes a lo Largo del Tiempo (Mensual)', fontsize=14)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Número de Accidentes', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Conclusiones:
# Se han reducido los accidentes el ultimo año y medio

#El mes de febrero durante los dos años ha tenido menos accidentalidad, esto puede deberse a que es el mes con menos días, al tener menos días, hay menos accidentes en el mes. Tenemos un dato atípico en agosto del 2019, esto puede deberse a que de ese mes no hubo datos de todos los días. Como se puede visualizar en la celda anterior, el último registro es el 24 de agosto del 2019.




# COMMAND ----------

# MAGIC %md
# MAGIC El gráfico de líneas que trazamos muestra claramente que no hay una tendencia alcista obvia en los accidentes a lo largo del tiempo.
# MAGIC
# MAGIC De la gráfica anterior, ¿qué meses parecen tener el menor número de accidentes? ¿Cuáles crees que son las razones detrás de esto?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exjercicio 3:
# MAGIC ¿Cómo varía el número de accidentes a lo largo de un solo día? Cree una nueva columna `HOUR` basada en los datos de la columna `TIME`, luego trace un gráfico de barras de la distribución por hora a lo largo del día.
# MAGIC
# MAGIC **Sugerencia:** Puede encontrar útil la función ```dt.hour```.

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

#Convierto a String
df['TIME'] = df['TIME'].astype(str)

#Usamos str.split() para extraer la hora directamente sin conversión a datetime
df['HOUR'] = df['TIME'].str.split(':').str[0].astype(int)

# Contar el número de accidentes por hora
accidents_by_hour = df.groupby('HOUR').size()

# Generar el gráfico de barras
plt.figure(figsize=(10, 6))
accidents_by_hour.plot(kind='bar', color='skyblue')

# Etiquetas y título
plt.title('Número de Accidentes a lo Largo del Día por Hora', fontsize=14)
plt.xlabel('Hora del Día', fontsize=12)
plt.ylabel('Número de Accidentes', fontsize=12)
plt.xticks(rotation=0)  # Para que las horas se muestren de forma horizontal
plt.grid(True)

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC En la pregunta anterior hemos agregado el número de accidentes por hora sin tener en cuenta la fecha y el lugar en que ocurrieron. ¿Qué crítica le daría a este enfoque?

# COMMAND ----------

# Respuesta:
# 1- A lo largo del dia, el numero de accidentes se va incrementando conforme más personas acuden a algun medio de transporte para realizar sus actividades llegando al pico de accidentes entre las 16 y 17 horas, coincidiendo esto con la hora salida de los trabajadores.

#2- Se debe segmentar mejor los datos para poder realizar el análisis correcto, en este caso la fecha y el lugar pueden suministrar informacion valiosa relacionada con algun fenomeno en particular que ocurre en dichos lugares y momento

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 4:
# MAGIC
# MAGIC ¿Cómo varía el número de accidentes en una sola semana? Trace un gráfico de barras basado en el recuento de accidentes por día de la semana.
# MAGIC
# MAGIC **Sugerencia:** Puede encontrar útil la función ```dt.weekday```.

# COMMAND ----------

# Solución propuesta

# Convertimos columna a tipo Date
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

# Extraer el día de la semana
df['DAY_OF_WEEK'] = df['DATE'].dt.weekday

# Contar el número de accidentes por día de la semana
accidents_by_day = df.groupby('DAY_OF_WEEK').size()

# Generar el gráfico de barras
plt.figure(figsize=(10, 6))
accidents_by_day.plot(kind='bar', color='lightcoral')

# Etiquetas y título
plt.title('Número de Accidentes por Día de la Semana', fontsize=14)
plt.xlabel('Día de la Semana', fontsize=12)
plt.ylabel('Número de Accidentes', fontsize=12)
plt.xticks(ticks=range(7), labels=['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'], rotation=0)
plt.grid(True)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 5:
# MAGIC
# MAGIC Trace una gráfica de barras del número total de accidentes en cada municipio, así como uno de los accidentes por milla cuadrada por municipio. ¿Qué puedes concluir?
# MAGIC
# MAGIC **Sugerencia:** Es posible que desee actualizar algunas de las claves en el diccionario del municipio para que coincidan con los nombres en el marco de datos.

# COMMAND ----------

# Solución propuesta

# Agrupar los accidentes por municipio y contar el número de accidentes
accidents_by_borough = df.groupby('BOROUGH').size()

#Diccionario de áreas de cada municipio (en millas cuadradas)

borough_area = {
    'MANHATTAN': 22.7,  # Área en millas cuadradas
    'BROOKLYN': 69.4,
    'QUEENS': 108.7,
    'BRONX': 42.2,
    'STATEN ISLAND': 57.5
}

#Calcular accidentes por milla cuadrada
accidents_by_borough = accidents_by_borough[accidents_by_borough.index.isin(borough_area.keys())]

# Calcular accidentes por milla cuadrada
accidents_per_sq_mile = accidents_by_borough / accidents_by_borough.index.map(borough_area)

# Paso 4: Generar los gráficos de barras

# Configurar el gráfico para el número total de accidentes por municipio
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico de barras del número total de accidentes por municipio
accidents_by_borough.plot(kind='bar', ax=ax1, color='lightcoral')
ax1.set_title('Número Total de Accidentes por Municipio')
ax1.set_xlabel('Municipio')
ax1.set_ylabel('Número de Accidentes')
ax1.set_xticklabels(accidents_by_borough.index, rotation=45)

# Gráfico de barras de accidentes por milla cuadrada por municipio
accidents_per_sq_mile.plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_title('Accidentes por Milla Cuadrada por Municipio')
ax2.set_xlabel('Municipio')
ax2.set_ylabel('Accidentes por Milla Cuadrada')
ax2.set_xticklabels(accidents_per_sq_mile.index, rotation=45)

# Ajustar el layout
plt.tight_layout()

# Mostrar los gráficos
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos ver que Brooklyn y Queens tienen un número muy alto de accidentes en relación con los otros tres condados. Pero, ¿qué tal por milla cuadrada?

# COMMAND ----------

# Conclusión:

#  1- Cuando se realiza el conteo de los datos se observa que brooklyn es el municipio con mayor numero de accidentes 
#2- Al tener en cuenta cuantos accidentes ocurren por milla cuadrada, se evidencia que manhattan es el municipio que presenta una mayor densidad de accidentes, siendo esta una medida más acertada a tener en cuenta.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 6:
# MAGIC
# MAGIC ¿Qué horas tienen más accidentes en cada municipio? Trace un gráfico de barras para cada municipio que muestre el número de accidentes por cada hora del día.
# MAGIC
# MAGIC **Sugerencia:** Puede usar ```sns.FacetGrid``` para crear una cuadrícula de parcelas con los datos por hora de cada municipio.

# COMMAND ----------

# Solución propuesta
import numpy as np
import seaborn as sns
#Agrupamos por municipio y por hora 
accidentes_por_municipio_hora = df.groupby(['BOROUGH', 'HOUR']).size().reset_index(name='Accidentes')

# Graficamos los resultados
plt.figure(figsize=(16, 6))

g = sns.FacetGrid(accidentes_por_municipio_hora, col='BOROUGH', col_wrap=2, height=4, aspect=2)
g.map(sns.barplot, 'HOUR', 'Accidentes')

g.set_axis_labels("Hora del Día", "Número de Accidentes")
g.fig.suptitle("Accidentes por Hora del Día por Municipio", y=1.02)

# COMMAND ----------

# MAGIC %md
# MAGIC **¿Es mayor el número de accidentes en diferentes momentos en diferentes distritos? ¿Deberíamos concentrarnos en diferentes momentos para cada municipio?**

# COMMAND ----------

#Respuesta:

#Para cada uno de los municipios se puede observar que los accidentes ocurren con mas frecuencia entre las 16-17 horas por lo cual no es completamente
#necesario centrance en atacar el problema para diferentes horas de cada municipio

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 7:
# MAGIC
# MAGIC ¿Qué factores provocan la mayoría de los accidentes? Evite contar dos veces los factores que contribuyen a un solo accidente.
# MAGIC
# MAGIC **Sugerencia:** Una forma de lidiar con las repeticiones es concatenar las columnas correspondientes conservando sus índices, puede hacerlo con las funciones ```pd.concat()``` y ```reset_index()```. Luego, use un ```group_by``` apropiado para contar el número de repeticiones de factores contribuidos por accidente.

# COMMAND ----------

# Solución propuesta

# Concatenar las columnas de factores contribuyentes
factors = pd.concat([df['CONTRIBUTING FACTOR VEHICLE 1'],
                     df['CONTRIBUTING FACTOR VEHICLE 2'],
                     df['CONTRIBUTING FACTOR VEHICLE 3'],
                     df['CONTRIBUTING FACTOR VEHICLE 4'],
                     df['CONTRIBUTING FACTOR VEHICLE 5']], axis=0)

# Limpiar los valores nulos
factors = factors.dropna()

# Contar la frecuencia de cada factor
factor_counts = factors.value_counts()

# Visualizar los resultados
# Seleccionamos los 10 factores más comunes
top_factors = factor_counts.head(10)

# Graficar los resultados
plt.figure(figsize=(10, 6))
top_factors.plot(kind='bar', color='lightblue')
plt.title('Factores que Provocan la Mayoría de los Accidentes', fontsize=16)
plt.xlabel('Factor Contribuyente', fontsize=12)
plt.ylabel('Número de Accidentes', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

display(factor_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 8:
# MAGIC
# MAGIC ¿Qué tipos de vehículos están más involucrados en accidentes por municipio? Evite contar dos veces el tipo de vehículos presentes en un solo accidente.
# MAGIC
# MAGIC **Sugerencia:** Puede aplicar un enfoque similar al utilizado en la pregunta anterior.

# COMMAND ----------

# Solución propuesta
# Concatenar las columnas de tipos de vehículos
vehicle_types = pd.concat([df['VEHICLE TYPE CODE 1'],
                           df['VEHICLE TYPE CODE 2'],
                           df['VEHICLE TYPE CODE 3'],
                           df['VEHICLE TYPE CODE 4'],
                           df['VEHICLE TYPE CODE 5']], axis=0)

# Limpiar los valores nulos 
vehicle_types = vehicle_types.dropna()

# Paso 3: Agrupar por municipio y contar los tipos de vehículos
vehicle_counts = pd.concat([df['BOROUGH'].reset_index(drop=True), vehicle_types.reset_index(drop=True)], axis=1)
vehicle_counts.columns = ['BOROUGH', 'VEHICLE TYPE']
vehicle_counts = vehicle_counts.groupby(['BOROUGH', 'VEHICLE TYPE']).size().reset_index(name='accidents')

# Visualización
# Crear gráficos de barras para cada municipio
boroughs = vehicle_counts['BOROUGH'].unique()
n_boroughs = len(boroughs)

fig, axes = plt.subplots(n_boroughs, 1, figsize=(10, 6 * n_boroughs))

# Si solo hay un municipio, `axes` no es un array, así que nos aseguramos de que sea una lista
if n_boroughs == 1:
    axes = [axes]

# Dibujar un gráfico de barras por municipio
for i, borough in enumerate(boroughs):
    borough_data = vehicle_counts[vehicle_counts['BOROUGH'] == borough]
    
    # Graficar los tipos de vehículos por número de accidentes
    borough_data.plot(kind='bar', x='VEHICLE TYPE', y='accidents', ax=axes[i], color='lightgreen')
    
    # Añadir título y etiquetas
    axes[i].set_title(f'Tipos de Vehículos Involucrados en Accidentes - {borough}', fontsize=14)
    axes[i].set_xlabel('Tipo de Vehículo', fontsize=12)
    axes[i].set_ylabel('Número de Accidentes', fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)  # Rotar las etiquetas de los tipos de vehículos

# Añadir un título global
plt.suptitle('Tipos de Vehículos Más Involucrados en Accidentes por Municipio', fontsize=16)

# Ajustar el diseño de la figura
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Asegura que el título global no se sobreponga

# Mostrar el gráfico
plt.show()

display(vehicle_counts)


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Concatenar las columnas de tipos de vehículos
vehicle_types = pd.concat([df['VEHICLE TYPE CODE 1'],
                           df['VEHICLE TYPE CODE 2'],
                           df['VEHICLE TYPE CODE 3'],
                           df['VEHICLE TYPE CODE 4'],
                           df['VEHICLE TYPE CODE 5']], axis=0)

# Limpiar los valores nulos
vehicle_types = vehicle_types.dropna()

# Paso 3: Agrupar por municipio y contar los tipos de vehículos
vehicle_counts = pd.concat([df['BOROUGH'].reset_index(drop=True), vehicle_types.reset_index(drop=True)], axis=1)
vehicle_counts.columns = ['BOROUGH', 'VEHICLE TYPE']
vehicle_counts = vehicle_counts.groupby(['BOROUGH', 'VEHICLE TYPE']).size().reset_index(name='accidents')

# Paso 4: Filtrar los 5 tipos de vehículos con más accidentes por municipio
top_5_vehicles_per_borough = vehicle_counts.groupby('BOROUGH').apply(lambda x: x.nlargest(5, 'accidents')).reset_index(drop=True)

# Visualización
# Crear gráficos de barras para cada municipio
boroughs = top_5_vehicles_per_borough['BOROUGH'].unique()
n_boroughs = len(boroughs)

fig, axes = plt.subplots(n_boroughs, 1, figsize=(10, 6 * n_boroughs))

# Si solo hay un municipio, `axes` no es un array, así que nos aseguramos de que sea una lista
if n_boroughs == 1:
    axes = [axes]

# Dibujar un gráfico de barras por municipio
for i, borough in enumerate(boroughs):
    borough_data = top_5_vehicles_per_borough[top_5_vehicles_per_borough['BOROUGH'] == borough]
    
    # Graficar los tipos de vehículos por número de accidentes
    borough_data.plot(kind='bar', x='VEHICLE TYPE', y='accidents', ax=axes[i], color='lightgreen')
    
    # Añadir título y etiquetas
    axes[i].set_title(f'Tipos de Vehículos Involucrados en Accidentes - {borough}', fontsize=14)
    axes[i].set_xlabel('Tipo de Vehículo', fontsize=12)
    axes[i].set_ylabel('Número de Accidentes', fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)  # Rotar las etiquetas de los tipos de vehículos

# Añadir un título global
plt.suptitle('Tipos de Vehículos Más Involucrados en Accidentes por Municipio', fontsize=16)

# Ajustar el diseño de la figura
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Asegura que el título global no se sobreponga

# Mostrar el gráfico
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 9:
# MAGIC
# MAGIC En 2018 para una [entrevista](https://www.nytimes.com/2019/01/01/nyregion/traffic-deaths-decrease-nyc.html) con The New York Times, el alcalde de Blasio de Nueva York declaró que *'Vision Zero está funcionando claramente'*. Ese año, el número de muertes en accidentes de tráfico en Nueva York se redujo a un histórico 202. Sin embargo, según lo informado por [am New York Metro](https://www.amny.com/news/vision-zero-de-blasio- 1-30707464 /), el número de víctimas mortales ha aumentado un 30% en el primer trimestre de 2019 en comparación con el año anterior y el número de peatones y ciclistas heridos no ha experimentado ninguna mejora.
# MAGIC
# MAGIC ¿Cómo utilizaría los datos proporcionados para comprender qué salió mal en el primer trimestre de 2019?
# MAGIC
# MAGIC
# MAGIC > - [ ] Considere los accidentes del primer trimestre de 2019. Luego, busque las causas más comunes de accidentes en los que estuvieron involucrados peatones y ciclistas. Dé una recomendación basada únicamente en esta información.
# MAGIC
# MAGIC > - [ ] Cree un par de mapas de calor de los accidentes que involucraron a peatones y ciclistas lesionados / muertos en el primer trimestre de 2018 y 2019. Compare estos dos para ver si hay algún cambio en la concentración de accidentes. En áreas críticas, estudie el tipo de factores involucrados en los accidentes. Dé una recomendación para visitar estas áreas para estudiar más el problema.   
# MAGIC
# MAGIC > - [ ] Los datos proporcionados son insuficientes para mejorar nuestra comprensión de la situación.
# MAGIC
# MAGIC > - [ ] Ninguna de las anteriores. Haría lo siguiente: *aquí tu respuesta recomendada*.

# COMMAND ----------

#Solucion Propuesta
df_first_case = df.copy()
import datetime

df_trim_2019 = df_first_case[ df_first_case['DATE']>=datetime.datetime(year=2019, month=1,day = 1)]
df_trim_2019 = df_trim_2019.reset_index()

df_trim = df_trim_2019[ df_trim_2019['DATE']<=datetime.datetime(year=2019, month=3,day = 30)].copy()
df_trim = df_trim.reset_index()

df_trim['only_pedestrian_cyclist'] = df_trim['NUMBER OF PEDESTRIANS INJURED'] +  df_trim['NUMBER OF PEDESTRIANS KILLED'] + df_trim['NUMBER OF CYCLIST INJURED'] + df_trim['NUMBER OF CYCLIST KILLED']
df_only = df_trim[df_trim['only_pedestrian_cyclist']!=0].copy()

factor_only_5 = df_only['CONTRIBUTING FACTOR VEHICLE 5']
factor_only_4 = df_only['CONTRIBUTING FACTOR VEHICLE 4']
factor_only_3 = df_only['CONTRIBUTING FACTOR VEHICLE 3']
factor_only_2 = df_only['CONTRIBUTING FACTOR VEHICLE 2']
factor_only_1 = df_only['CONTRIBUTING FACTOR VEHICLE 1']

only_factor = pd.concat([factor_only_1,factor_only_2,factor_only_3,factor_only_4,factor_only_5])
only_factor = only_factor.reset_index()

only_factor_aa = only_factor.groupby(only_factor.columns.tolist(),as_index=False).size()
only_factor_aag = only_factor_aa.groupby(0)['size'].count()
only_factor_aag = only_factor_aag.reset_index()
only_factor_aag_sin_unspecifed = only_factor_aag[only_factor_aag[0]!='Unspecified'].copy()
only_factor_aag_sin_unspecifed.rename(columns = {0:'CONTRIBUTING FACTOR','size':'ACCIDENTS'},inplace = True)
only_factor_aag_sin_unspecifed.sort_values(by=['ACCIDENTS'],ascending= False).head(3)

# Conclusión:

# Los datos muestran que para el primer trimestre de 2019 los accidentes en los cuales estan más involucrados los ciclistas y 
#peatones resultaron ser los relacionados a la falta de atencion por parte del conductor, no ceder al paso y fallos por parte de los mismos
#ciclistas y peatones, esta tres razones estan muy correlacionadas por lo cual se deben evaluar estrategias que mitigen los problemas de manera conjunta.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejercicio 10:
# MAGIC
# MAGIC Calcula el número de muertes provocadas por cada tipo de vehículo. Trace un gráfico de barras para los 5 vehículos principales. ¿Qué vehículos están involucrados con mayor frecuencia en las muertes y cuánto más que los demás?
# MAGIC
# MAGIC **Por ejemplo,** si dos personas murieron en un accidente en el que estuvieron involucrados 5 vehículos: 4 son VEHÍCULOS DE PASAJEROS y 1 es un VAGÓN DEPORTIVO / ESTACIÓN. Luego, agregaríamos dos muertes a cada tipo de VEHÍCULO DE PASAJEROS y VAGÓN DE ESTACIÓN / SERVICIO DEPORTIVO.
# MAGIC
# MAGIC **Sugerencia:** Es posible que desee crear una nueva columna con el número total de muertes en el accidente. Para eso, puede encontrar útil la función ```.to_numpy()```. Luego, proceda como los ejercicios anteriores para evitar contabilizar dos veces el tipo de vehículos.

# COMMAND ----------

# Solución propuesta

#Creamos la columna con el número total de muertes por accidente
df["MUERTES_TOTALES"] = df[[ "NUMBER OF PEDESTRIANS KILLED", "NUMBER OF CYCLIST KILLED", "NUMBER OF MOTORIST KILLED"]].sum(axis=1)

vehiculo_columnas = ["VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2","VEHICLE TYPE CODE 3","VEHICLE TYPE CODE 4","VEHICLE TYPE CODE 5"]

vehiculos_muertes = df.melt(id_vars=["MUERTES_TOTALES"], value_vars=vehiculo_columnas,value_name="VEHICULO") 

#Quitamos los vehículos desconocidos
vehiculos_muertes = vehiculos_muertes[vehiculos_muertes["VEHICULO"].str.lower() != "unknown"]

#Sumamos muertes por tipo de vehículo sin repetir datos
muertes_por_vehiculo = vehiculos_muertes.groupby("VEHICULO")["MUERTES_TOTALES"].sum().sort_values(ascending=False)

top_5_vehiculos = muertes_por_vehiculo.head(5)

# Graficamos
plt.figure(figsize=(10, 5))
sns.barplot(x=top_5_vehiculos.index, y=top_5_vehiculos.values)
plt.xticks(rotation=45)
plt.xlabel("Tipo de Vehículo")
plt.ylabel("Número de Muertes")
plt.title("Top 5 Vehículos con más Muertes en Accidentes")
plt.show()
