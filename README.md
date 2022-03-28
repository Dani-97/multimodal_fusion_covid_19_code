# Código del proyecto de machine learning aplicado a datos de COVID-19

## Ficheros ejecutables

- **analyze_input_dataset.py:** este fichero ejecutable se utiliza para analizar la información del dataset original, sin ningún tipo de procesamiento de los datos. Entre otras cosas, permite obtener los histogramas de la distribución de los valores de cada uno de los atributos de la tabla.

- **analyze_associations.py:** este fichero ejecutable se emplea para analizar el fichero que contiene las asociaciones entre el nombre de las imágenes del dataset y el código que tienen asociado en la tabla CSV del CHUAC. 

- **analyze_built_datasets.py:** este fichero ejecutable se utiliza para analizar la información de los datasets construidos a partir de la tabla original.

- **build_dataset.py:** este fichero ejecutable se utiliza para construir los datasets a partir de la tabla original pero sin incluir en ningún caso deep features. Se consideró que era muy complejo meter el caso de uso que solo obtiene tablas con datos clínicos en el mismo script que el que implementa la construcción de datasets que también incluyen datos de deep features, por lo que cada aproximación tiene su propio script.

- **build_dataset_with_deep_features.py:** este fichero ejecutable se utiliza para construir los datasets extrayendo deep features de imágenes radiológicas.

- **dataset_queries.py:** este fichero permite realizar queries contra el dataset. Esto se  consigue dado que el fichero CSV es leído por el script y convertido a un DataFrame de pandas. 

- **train.py:** este fichero ejecutable permite entrenar un determinado modelo de aprendizaje máquina con los datos de un dataset construído.

## Ficheros de utilidad

- **utils.py:** este fichero no es ejecutable y contiene algunas funciones de utilidad interesantes para facilitar la implementación de otros scripts.

**FICHEROS DE UTILIDAD PARA EL ANÁLISIS DE LOS DATOS**

- **analysis/utils_analysis_built_datasets.py:** fichero para el análisis de los datos de los datasets construidos a partir del dataset original.
- **analysis/utils_histograms.py:** fichero de utilidad para la creación de histogramas que muestren la distribución de los valores de los atributos en la tabla original.
- Los ficheros **analysis/ad_hoc_items_original_dataset.cfg** y **analysis/ad_hoc_items_reduced_dataset.cfg** contienen datos ad-hoc necesarios para la obtención de los histogramas de los diferentes atributos.

**FICHEROS DE UTILIDAD PARA LOS DATASETS**

- **datasets/utils_build_datasets.py:** clases que contienen las diferentes implementaciones que existen para construir datasets.
- **datasets/utils_balancing.py:** clases para el balanceo de los datos (oversampling, undersampling, no balanceo...).
- **datasets/utils_features.py:** clases que incluyen las llamadas a métodos de selección y extracción de características.
- **datasets/utils_datasets.py:** clases que contienen las diferentes aproximaciones para dividir los datos en particiones (Holdout, Cross Validation, Bootstraping o cualquier aproximación que se considere conveniente).
- El fichero **list/list_without_weird_rows.cfg** contiene una lista que especifica el código de todas aquellas filas que no tengan un tiempo de urgencia demasiado grande (almacenada en binario utilizando pickle). Cabe destacar que el criterio para decidir si un tiempo de urgencia es muy grande o no, puede ser variable. 

**FICHEROS DE UTILIDAD PARA LOS CLASIFICADORES**

- **classifiers/utils_classifiers.py:** este fichero contiene las clases que implementan las diferentes funcionalidades necesarias para los clasificadores.

**FICHEROS DE UTILIDAD PARA LOS MODELOS DE REGRESIÓN**

- **regressors/utils_regressors.py:** este fichero contiene las clases que implementan las diferentes funcionalidades necesarias para los modelos de regresión.

**FICHEROS DE UTILIDAD PARA LOS MODELOS DE OBTENCIÓN DE DEEP FEATURES**

- **deep_features/utils_architectures.py:** este fichero contiene las clases que definen los diferentes clasificadores utilizados para la obtención de deep features, con todas las funcionalidades que son necesarias.

## Ficheros de extensión .sh

En estos ficheros se incluyen las llamadas a scripts de Python, tanto a modo de prueba (como es el caso de train_debugging.sh) como llevando a cabo tareas concretas utilizando clasificadores y extractores de características concretos.

# Detalles importantes del código de programación de este proyecto

El código de programación de este proyecto está creado con la intención de respetar algunos patrones de diseño del software básicos y así facilitar el mantenimiento del mismo y el desarrollo de nuevas funcionalidades. En este README se describe brevemente el contenido de los ficheros y los directorios del proyecto.

## Ficheros de extensión .py

Con respecto a los ficheros .py, existen principalmente 2 tipos:

- Los ficheros ejecutables, que disponen de un main y que, por tanto, se pueden ejecutar directamente.
- Los ficheros de código utilizable que no disponen de un main y que, por tanto, no se pueden ejecutar por sí solos.

**Ficheros ejecutables:** estos ficheros implementan los diferentes casos de uso de la aplicación que se encuentra en este proyecto:

- *Caso de uso 1*: **Análisis del dataset original**. Este caso de uso permite realizar diferentes análisis de la tabla original del CHUAC, es decir, sin ningún tipo de procesamiento previo. Una de las funcionalidades que proporciona este análisis es la creación de los histogramas que muestran la distribución de los valores de los diferentes atributos incluidos en el dataset.

- *Caso de uso 2*: **Construcción de datasets**. Este caso de uso permite crear, a partir del dataset original del CHUAC, un nuevo dataset ya con todos los campos procesados, es decir, preparado para entrenar los diferentes modelos de aprendizaje (entre otras cosas, cambiando todos los datos a formato numérico, incluidas las variables categóricas y determinando las salidas que el sistema debe aprender).

- *Caso de uso 3*: **Entrenamiento de los modelos de aprendizaje máquina**. Para este caso de uso, una vez que los datasets han sido construídos por medio de la implementación del caso de uso anterior, se procede a entrenar los modelos de aprendizaje máquina.

**Ficheros no ejecutables:** estos ficheros de Python no pueden ejecutarse por sí mismos pero contienen el código realmente importante de las ejecuciones. Por ejemplo, hay un fichero de utilidades dentro de la carpeta "classifiers" que se llama "utils_classifiers.py" donde se crean los clasificadores y donde se especifican sus correspondientes métodos de entrenamiento, de test y todos aquellos que sean necesarios. Cuando desde un fichero ejecutable se quiere entrenar un modelo y validarlo, lo que se hace es:

**1. Crear el modelo.**

**2. Llamar a la función de entrenamiento.**

**3. Llamar posteriormente a la función de test sobre ese modelo entrenado para obtener la salida correspondiente.**

**4. Llamar a la función para obtener las métricas de evaluación. **

Por tanto, en resumen, el fichero ejecutable es el que crea los objetos y llama a los métodos pero es el código de las clases implementadas en los ficheros de utilidad el que hace el trabajo realmente importante.

## Uso de los scripts del proyecto

Todos los scripts ejecutables utilizan una librería para parsear los argumentos que tiene una opción de ayuda por defecto. Cada parámetro está convenientemente descrito en esa opción de ayuda que se puede obtener ejecutando "python3 nombre_del_script.py -h". Del mismo modo, los ficheros .sh disponibles en el proyecto sirven de referencia para tener un ejemplo sobre como llamar a los scripts ejecutables.

## ¿Cómo se hace que los scripts sean ampliamente configurables?

Como se describe al principio de este README, este software se ha diseñado para que sea lo más mantenible posible y que añadir nuevas funcionalidades sea relativamente sencillo. Para conseguir esto, se utiliza una clase denominada "UniversalFactory" que permite instanciar un objeto de cualquier clase con solo pasarle su nombre como un string y los parámetros que necesita el constructor.

Para que la creación de objetos con la factoría universal sea mucho más inmediata, se utiliza el patrón adaptador. Por ejemplo, en el caso de los clasificadores, cada clasificador se encapsula en una clase que tiene la denominación del propio clasificador (por ejemplo, SVM en el caso de la máquina de soporte vectorial o DT en el caso de los árboles de decisión) con una sintaxis del estilo X_Classifier teniendo, por tanto, clases como SVM_Classifier ó DT_Classifier. Por tanto, tienen que respetar esa sintaxis del nombre ya que, de lo contrario, no funcionará.
