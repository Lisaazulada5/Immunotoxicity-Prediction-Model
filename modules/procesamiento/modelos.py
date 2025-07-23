import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import statsmodels.api as sm

# modelos.py

import pandas as pd
from sklearn.model_selection import train_test_split

"""
Dividir los datos
"""

from modules.procesamiento.limpieza_datos import guardar_csv

def dividir_datos(df, columna_etiqueta, test_size=0.2, random_state=42):
    """
    Función para dividir el DataFrame en conjunto de entrenamiento y conjunto de prueba.
    Guarda los conjuntos de datos como CSV.

    :param df: DataFrame con los datos
    :param columna_etiqueta: Nombre de la columna que contiene las etiquetas (e.g., 'etiqueta')
    :param test_size: Proporción de datos para el conjunto de prueba (por defecto 20%)
    :param random_state: Semilla para garantizar la reproducibilidad (por defecto 42)
    :return: None
    """
    # Dividir el DataFrame en características (X) y etiquetas (y)
    X = df.drop(columna_etiqueta, axis=1)  # Características
    y = df[columna_etiqueta]  # Etiquetas (por ejemplo, activa/inactiva)

    # Dividir en entrenamiento y prueba, manteniendo la proporción de las clases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Unir X_train, X_test con y_train, y_test de nuevo en DataFrames para exportar
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Guardar los conjuntos de datos como archivos CSV
    guardar_csv(train_df, 'data/train_data.csv')
    guardar_csv(test_df, 'data/test_data.csv')

    print("Los conjuntos de datos se han dividido y guardado correctamente como 'train_data.csv' y 'test_data.csv'.")





"""
Regresión  multiple
"""


def modelo_regresion(df, X_columns, Y_column, significance_level=0.05): #Esta función aplica la eliminación hacia atrás para seleccionar las variables más significativas en un modelo de regresión.
    X = df[X_columns].values #Extrae los valores de las columnas independientes (variables predictoras) del DataFrame.
    Y = df[Y_column].values #Extrae los valores de la columna dependiente (variable objetivo) del DataFrame.

    X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1) #Agrega una columna de unos al inicio de la matriz X para incluir el término de intercepto en la regresión.
    X_opt = X.copy()
    regressor_OLS = sm.OLS(Y, X_opt).fit() #Crea y ajusta un modelo de regresión lineal usando mínimos cuadrados ordinarios (OLS)

    return regressor_OLS #Devuelve el modelo final después de eliminar las variables no significativas

"""
Regresión logistica
"""

# modelos.py
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def regresion_logistica(df, columnas, target):
    """
    Función para realizar regresión logística.

    :param df: DataFrame con los datos.
    :param columnas: Lista de nombres de columnas para las variables predictoras.
    :param target: Nombre de la columna de la variable dependiente.
    :return: Modelo ajustado y resultados de predicción.
    """
    X = df[columnas]
    y = df[target]

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Agregar la constante para el intercepto en el modelo
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Crear el modelo de regresión logística
    log_reg = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
    log_reg.fit(X_train, y_train)

    # Predicciones
    y_pred = log_reg.predict(X_test)
    y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Mostrar los resultados
    print(f"Precisión del modelo: {accuracy:.4f}")
    print("Matriz de confusión:")
    print(cm)
    print("Reporte de clasificación:")
    print(report)

    return log_reg, accuracy, cm, report,y_test, y_pred_proba

"""
COEFICIENTES DE LA REGRESION LOGISTICA
"""
import pandas as pd

def obtener_coeficientes(columnas, modelo):
    """
    Función para obtener los coeficientes del modelo y asociarlos con las variables predictoras.

    :param columnas: Lista de nombres de las variables predictoras.
    :param modelo: El modelo entrenado (por ejemplo, un modelo de regresión logística).
    :return: DataFrame con los coeficientes y las variables asociadas.
    """
    # Extraer los coeficientes del modelo
    coeficientes = modelo.params  # Extraer coeficientes para la clasificación binomial

    # Crear un DataFrame con las variables y sus coeficientes
    coef_df = pd.DataFrame({
        'Variable': ['Coeficiente de Intercepto'] + columnas,
        'Coeficiente': coeficientes
    })

    return coef_df

"""
MODELO DE ÁRBOLES DE DECISION
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def arbol_decision(X, y, test_size=0.2, random_state=42):
    """
    Entrena un modelo de Árbol de Decisión y lo evalúa con datos de prueba.

    Parámetros:
    - X: DataFrame o array con las características.
    - y: Serie o array con la variable objetivo.
    - criterio: 'gini' o 'entropy' para la función de evaluación del árbol.
    - max_depth: Profundidad máxima del árbol (None para sin restricción).
    - min_samples_split: Mínimo de muestras para dividir un nodo.
    - min_samples_leaf: Mínimo de muestras en una hoja.
    - test_size: Proporción de datos usados para prueba (por defecto 20%).
    - random_state: Semilla para la reproducibilidad.

    Retorna:
    - modelo: El modelo entrenado.
    - accuracy_train: Precisión en el conjunto de entrenamiento.
    - accuracy_test: Precisión en el conjunto de prueba.
    - cm_test: Matriz de confusión en el conjunto de prueba.
    - report_test: Reporte de clasificación en el conjunto de prueba.
    """
    # Separar datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, random_state=random_state, stratify=y)

    # Definir y entrenar el modelo
    modelo = DecisionTreeClassifier(criterion= 'entropy', max_depth=3,
                                    min_samples_split=2,
                                    min_samples_leaf=10,
                                    random_state=42, class_weight='balanced'
                                    ,ccp_alpha=0.01)

    modelo.fit(X_train, y_train)

    # Evaluación en entrenamiento
    y_pred_train = modelo.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    # Evaluación en prueba
    y_pred_test = modelo.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    report_test = classification_report(y_test, y_pred_test)
    y_pred_prob_test = modelo.predict_proba(X_test)[:, 1]

    print(f"Precisión en entrenamiento: {accuracy_train:.4f}")
    print(f"Precisión en prueba: {accuracy_test:.4f}")
    print("Matriz de confusión (Prueba):")
    print(cm_test)
    print("Reporte de clasificación (Prueba):")
    print(report_test)

    # Validación cruzada
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    auc_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc")

    print("\n______________________________")
    print("CROSS VALIDATION AUC-ROC")
    print("______________________________")
    print(f"AUC en cada fold: {auc_scores}")
    print(f"AUC promedio: {np.mean(auc_scores)}")
    print(f"Desviación estándar: {np.std(auc_scores)}")

    return modelo, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_test, y_test


""""
REGLOG LOGIT CON SUMMARY PARA VER  VALOR P DE COEFICIENTES
"""

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def regresion_logistica_sm(df, columnas, target):
    """
    Función para realizar regresión logística utilizando statsmodels,
    obteniendo el summary con los p-valores de los coeficientes.

    :param df: DataFrame con los datos.
    :param columnas: Lista de nombres de columnas para las variables predictoras.
    :param target: Nombre de la columna de la variable dependiente.
    :return: Modelo ajustado, summary, precisión, matriz de confusión, reporte de clasificación,
             datos reales de test (y_test) y probabilidades predichas (y_pred_proba).
    """
    # Extraer las variables predictoras y la variable dependiente
    X = df[columnas]
    y = df[target]

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Agregar la constante para el intercepto en el modelo
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Ajustar el modelo de regresión logística usando Logit de statsmodels
    modelo = sm.Logit(y_train, X_train).fit()

    # Obtener y mostrar el summary del modelo (incluye p-valores de los coeficientes)
    summary = modelo.summary()
    print(summary)

    # Realizar predicciones sobre el conjunto de prueba
    y_pred_proba = modelo.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)  # Clasificación con umbral 0.5

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Precisión del modelo: {accuracy:.4f}")
    print("Matriz de confusión:")
    print(cm)
    print("Reporte de clasificación:")
    print(report)

    return modelo, summary, accuracy, cm, report, y_test, y_pred_proba


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


def entrenar_random_forest(X, y, n_estimators=100, random_state=42):
    """
    Entrena un modelo Random Forest y muestra métricas de desempeño.
    :param X: DataFrame con las características
    :param y: Serie con la variable objetivo
    :param n_estimators: Número de árboles en el bosque
    :param random_state: Semilla aleatoria para reproducibilidad
    """
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Inicializar y entrenar el modelo
    modelo = RandomForestClassifier(criterion='gini',  # Cálculo de la impureza
    min_samples_leaf=1,  # Mínimo de muestras por hoja
    min_samples_split=5,  # Mínimo de muestras para dividir un nodo
    n_estimators=300,  # Número de árboles
    random_state=random_state,  # Semilla para reproducibilidad
    max_depth=5, class_weight='balanced') #,  ccp_alpha=  0.01) # Profundidad máxima del árbol

    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    y_pred_prob_test = modelo.predict_proba(X_test)[:, 1]

    # Evaluación
    #print("\n______________________________")
    #print("RANDOM FOREST")
    print("______________________________")
    print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.4f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # Validación cruzada
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    auc_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc")

    print("\n______________________________")
    print("CROSS VALIDATION AUC-ROC")
    print("______________________________")
    print(f"AUC en cada fold: {auc_scores}")
    print(f"AUC promedio: {np.mean(auc_scores)}")
    print(f"Desviación estándar: {np.std(auc_scores)}")

    return modelo, cm,y_pred_prob_test , y_test

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, make_scorer, f1_score

import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, make_scorer, f1_score


"""
MODELOS XGBOOST
"""
def entrenar_xgboost(X, y, test_size=0.2, random_state=42, cv=5):
    """
    Entrena un modelo XGBoost con scale_pos_weight y realiza validación cruzada.

    Parámetros:
    - X: Features (DataFrame o array)
    - y: Labels (array)
    - test_size: Proporción de datos para prueba
    - random_state: Semilla para reproducibilidad
    - cv: Número de folds para validación cruzada

    Retorna:
    - modelo: Modelo XGBoost entrenado
    """

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


    # Calcular scale_pos_weight (relación entre clases)
    #peso_clase = sum(y_train == 0) / sum(y_train == 1)
    #print(f"Peso de la clase minoritaria: {peso_clase:.2f}")

    # Configurar y entrenar modelo XGBoost
    modelo = xgb.XGBClassifier(colsample_bytree =1 ,
                               max_depth=10,
                               learning_rate=0.1,
                               n_estimators=100,
                               eval_metric="aucpr",
                               gamma = 0.1,
                               subsample = 0.7,
                               scale_pos_weight = 1.48)
                               #reg_alpha =0.0001907254857105348, reg_lambda = 1.4848771867376586e-05, scale_pos_weight = 1.98)
    modelo.fit(X_train, y_train)

    # Evaluación en test set
    y_pred = modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    y_pred_prob_test = modelo.predict_proba(X_test)[:, 1]

    # Validación cruzada
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    auc_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc")

    # 🔹 **Imprimir métricas dentro de la función**
    print("\n🔹 Reporte de Clasificación en Test Set:")
    for clase, valores in reporte.items():
        if isinstance(valores, dict):
            print(f"\nClase {clase}:")
            for metrica, valor in valores.items():
                print(f"  {metrica}: {valor:.4f}")

    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\n______________________________")
    print("CROSS VALIDATION AUC-ROC")
    print("______________________________")
    print(f"AUC en cada fold: {auc_scores}")
    print(f"AUC promedio: {np.mean(auc_scores)}")
    print(f"Desviación estándar: {np.std(auc_scores)}")

    return modelo, cm, y_pred_prob_test, y_test

"""
MAQUINAS DE SOPORTE
"""
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
import numpy as np
from imblearn.over_sampling import SMOTE


def entrenar_svm(X, y, test_size=0.2, random_state=42, cv=10):
    """
    Entrena un modelo SVM con kernel RBF, realiza validación cruzada y muestra métricas.

    Parámetros:
    - df: DataFrame con los datos
    - columnas_predictoras: Lista de columnas a usar como variables predictoras
    - columna_target: Nombre de la columna objetivo
    - test_size: Proporción de datos para test (default 0.2)
    - random_state: Semilla para reproducibilidad (default 42)
    - cv: Número de folds en validación cruzada (default 5)
    """
    # Separar variables predictoras y objetivo
    #X = df[columnas_predictoras]
    #y = df[columna_target]

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Definir y entrenar el modelo SVM con kernel RBF
    modelo = SVC(kernel='poly',
                 C=0.1,
                 gamma= "scale",
                 coef0 = 0.1,
                 degree = 4,
                 random_state=random_state,
                 probability=True,
                 class_weight= "balanced")
    # Validación cruzada
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc")

    # entrenar el modelo SVM con kernel RBF
    modelo.fit(X_train, y_train)

    # Predicción en test
    y_pred = modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    y_pred_prob_test = modelo.predict_proba(X_test)[:, 1]

    # Validación cruzada
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    auc_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc")

    # Imprimir métricas
    print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.4f}")
    print("Matriz de confusión")
    print(cm)
    print("🔹 Reporte de Clasificación en Test Set:")
    print(classification_report(y_test, y_pred))
    print("\n______________________________")
    print("CROSS VALIDATION AUC-ROC")
    print("______________________________")
    print(f"AUC en cada fold: {auc_scores}")
    print(f"AUC promedio: {np.mean(auc_scores)}")
    print(f"Desviación estándar: {np.std(auc_scores)}")
    return modelo, cm, y_pred_prob_test, y_test


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


"""
REDES NEURONALES
"""
def entrenar_red_neuronal(X, y, test_size=0.2, random_state=42, epochs=50, batch_size=32):
    """
    Entrena una red neuronal simple para clasificación binaria.

    Parámetros:
    - X: Datos de entrada (numpy array o DataFrame)
    - y: Etiquetas (numpy array o Series)
    - test_size: Proporción del conjunto de prueba
    - random_state: Semilla para reproducibilidad
    - epochs: Número de veces que la red verá los datos completos
    - batch_size: Tamaño de los lotes de entrenamiento
    """
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Escalado de datos (importante para redes neuronales)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Definir la arquitectura de la red neuronal
    modelo = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Capa de salida para clasificación binaria
    ])

    # Compilar el modelo
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    modelo.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    # Evaluar el modelo
    y_pred_prob = modelo.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convertir probabilidades en 0 o 1

    # Imprimir métricas
    print("🔹 Matriz de Confusión")
    print(confusion_matrix(y_test, y_pred))
    print("\n🔹 Reporte de Clasificación")
    print(classification_report(y_test, y_pred))

    return modelo


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def entrenar_knn(X, y, test_size=0.2, n_neighbors=5, random_state=42):
    """
    Entrena un modelo KNN con los datos proporcionados, dividiendo en entrenamiento y validación.

    Parámetros:
    - X: DataFrame o array con las características.
    - y: Serie o array con la variable objetivo (clase).
    - test_size: Proporción del conjunto de prueba (por defecto 0.2).
    - n_neighbors: Número de vecinos en KNN (por defecto 5).
    - random_state: Semilla para la reproducibilidad.

    Retorna:
    - Un diccionario con la matriz de confusión y el reporte de clasificación.
    """
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)

    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    # Entrenar modelo KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Predicciones
    y_pred = knn.predict(X_test)

    # Evaluación
    matriz_confusion = confusion_matrix(y_test, y_pred)
    reporte_clasificacion = classification_report(y_test, y_pred)

    return matriz_confusion, reporte_clasificacion


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV, StratifiedKFold


def entrenar_modelo(modelo, param_grid, X, y, cv=10, scoring='f1'):
    """
    Realiza GridSearchCV para encontrar los mejores hiperparámetros de un modelo.

    Parámetros:
    - modelo: modelo base de scikit-learn (ej. DecisionTreeClassifier()).
    - param_grid: diccionario con los hiperparámetros a evaluar.
    - X: Variables predictoras.
    - y: Variable objetivo.
    - cv: Número de folds para validación cruzada (default=10).
    - scoring: Métrica a optimizar (default='f1').

    Retorna:
    - Mejor modelo ajustado con los mejores hiperparámetros.
    """

    # Validación cruzada estratificada para evitar sobreajuste en clases desbalanceadas
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        modelo,
        param_grid,
        scoring=scoring,
        cv=skf,
        verbose=1,
        n_jobs=-1
    )

    # Ajustar el modelo
    grid_search.fit(X, y)

    # Mejor modelo encontrado
    mejor_modelo = grid_search.best_estimator_

    print("Mejores hiperparámetros:", grid_search.best_params_)

    return mejor_modelo


import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Función personalizada para calcular F1-Score
def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred = np.round(y_pred)  # Convertir probabilidades en clases binarias (0 o 1)
    return "f1", f1_score(y_true, y_pred)



# Función objetivo para la optimización bayesiana
import optuna
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Función personalizada para calcular F1-Score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np

# Función de evaluación personalizada para F1-score en XGBoost
def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred = np.round(y_pred)  # Convertir probabilidades en clases binarias (0 o 1)
    return "f1", f1_score(y_true, y_pred)

def evaluar_modelo(params, X, y):
    """Entrena y evalúa el modelo usando validación cruzada."""
    model = XGBClassifier(**params, use_label_encoder=False)

    scores = cross_val_score(model, X, y, cv=5, scoring="f1")  # Usando F1-score como métrica
    return np.mean(scores)  # Retorna el promedio de F1-score

def objective(trial, X, y):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),  # Tasa de aprendizaje (queda igual)
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),  # Fracción de datos usada en cada árbol
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # Fracción de features por árbol
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),  # Penalización en la creación de nodos
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),  # Regularización L1 más ajustada
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),  # Regularización L2 más ajustada
        "max_depth": trial.suggest_int("max_depth", 3, 12),  # Profundidad del árbol (queda igual)
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),  # Número de árboles (queda igual)
    }
    return evaluar_modelo(params, X, y)
