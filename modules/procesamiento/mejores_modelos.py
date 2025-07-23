import pandas as pd
from data.manejo_archivos import leer_csv, guardar_csv
from modules.procesamiento.modelos import entrenar_xgboost
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


"""
SOLO PARA ANALIZAR RAPIDAMENTE LAS SUSTANCIAS PERFLUORADAS
"""

"""
Analisis sustancias perfluoradas
"""

perfluoradas = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/perfluoradas.csv')
#borrar columnas repetidas
perfluoradas = perfluoradas.drop_duplicates(subset=['INCI Name/Substance Name'], keep='first')





# Definir las columnas predictoras y la variable objetivo
train_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_fingerprints.csv')

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS fisicoquimicas')
print('----------------')
train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS = train_data_fingerprints.iloc[:, 2054:2222]
MACCS = MACCS.add_prefix("MACCS_")
Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
ECFP = train_data_fingerprints.iloc[:, 6:2054]
ECFP = ECFP.add_prefix("ECFP_")
columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_ECFP_fisicoquimicas, cm, y_prob_ECFP_MACCS_FISICOQUIMICOS, y_test_ECFP_MACCS_FISICOQUIMICOS, X_train, X_test = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS +  Fisicoquimicas')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
test_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/test_data_fingerprints.csv')
test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
#test_data_fingerprints.rename(columns=lambda col: f"MACCS_{col}" if col in test_data_fingerprints.iloc[:, 2054:2222] else col, inplace=True)
MACCS_test = test_data_fingerprints.iloc[:, 2054:2222]
MACCS_test = MACCS_test.add_prefix("MACCS_")
Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
ECFP_test = ECFP_test.add_prefix("ECFP_")
columns_subset_test = pd.concat([ECFP_test, MACCS_test, Fisicoquimicas_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_ECFP_fisicoquimicas.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS']
# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusión:")
print(cm)
#Cálculo de métricas
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Mostrar las métricas
print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Exactitud: {accuracy}")


import shap

"""
SHAP MODELO ENTRENADO
"""
explainer = shap.TreeExplainer(xgboost_MACCS_ECFP_fisicoquimicas, X_train)
shap_values = explainer(X_test)  # Sin la columna de predicción
# Crear el gráfico SHAP con título

# Crear la figura y agregar título
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)  # Desactiva el show automático de SHAP
plt.title("SHAP Modelo dataset desbalanceado + fisicoquímicas", fontsize=14, pad=30)  # Aumenta el pad
# Ajustar la posición del gráfico
plt.subplots_adjust(top=0.85)  # Reduce este valor para bajar más el gráfico
# Guardar la figura
#plt.savefig('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/graficas/shap_modelo.png', dpi=300)
#plt.show()  # Mostrar la figura en pantalla






"""
PREDICCION DEL MODELO MOLECULAS COSING
"""
X_nuevo = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS.csv')
#X_nuevo = X_nuevo.drop_duplicates(subset=['SMILES'], keep='first')
#X_nuevo = X_nuevo.drop(columns=["SMILES", "ECFP", "MACCS"])
#probs = xgboost_MACCS_ECFP_fisicoquimicas.predict_proba(X_nuevo)
#preds = xgboost_MACCS_ECFP_fisicoquimicas.predict(X_nuevo)

PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES =  X_nuevo.copy()
PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES["Predicción"] = xgboost_MACCS_ECFP_fisicoquimicas.predict(X_nuevo)
PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES[["Probabilidad_Inac", "Probabilidad_Act"]] = xgboost_MACCS_ECFP_fisicoquimicas.predict_proba(X_nuevo)
guardar_csv(PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES, 'C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES.csv')
print("realizado")

import shap

"""
SHAP MODELO PREDICCIONES
"""
explainer = shap.TreeExplainer(xgboost_MACCS_ECFP_fisicoquimicas)
shap_values = explainer(X_nuevo)  # Sin la columna de predicción
# Crear el gráfico SHAP con título

# Crear la figura y agregar título
plt.figure()
shap.summary_plot(shap_values, X_nuevo, show=False)  # Desactiva el show automático de SHAP
plt.title("SHAP Predicciones", fontsize=14, pad=30)  # Aumenta el pad
# Ajustar la posición del gráfico
plt.subplots_adjust(top=0.85)  # Reduce este valor para bajar más el gráfico
# Guardar la figura
plt.savefig('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/graficas/shap_modelo_Predicciones.png', dpi=300)
plt.show()  # Mostrar la figura en pantalla

"""
Dependence SHAP
"""
shap_values_array = shap_values.values  # Extraer los valores SHAP
#shap.dependence_plot("LogP_scaled", shap_values_array, X_nuevo)
#shap.dependence_plot("Peso_Molecular_scaled", shap_values_array, X_nuevo)
#shap.dependence_plot("TPSA_scaled", shap_values_array, X_nuevo)
#shap.dependence_plot("NumRotatableBonds_scaled", shap_values_array, X_nuevo)
#shap.dependence_plot("Dobles_Enlaces_scaled", shap_values_array, X_nuevo)
#shap.dependence_plot("MACCS_154.1", shap_values_array, X_nuevo)
#shap.dependence_plot("MACCS_99.1", shap_values_array, X_nuevo)
#shap.dependence_plot("MACCS_49.1", shap_values_array, X_nuevo)
shap.dependence_plot("MACCS_44.1", shap_values_array, X_nuevo)
#shap.dependence_plot("MACCS_98.1", shap_values_array, X_nuevo)
#shap.dependence_plot("MACCS_109.1", shap_values_array, X_nuevo)
#shap.dependence_plot("MACCS_72.1", shap_values_array, X_nuevo)
#shap.dependence_plot("ECFP_1873", shap_values_array, X_nuevo)
shap.dependence_plot("MACCS_109.1", shap_values_array, X_nuevo)
shap.dependence_plot("MACCS_98.1", shap_values_array, X_nuevo)






"""
Segundo mejor modelo
"""

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS fisicoquimicas')
print('----------------')
train_data_scaled_electronic = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_scaled_electronic.csv')
#print(train_data_scaled_electronic)
propiedades_electronicas_a_usar = ["PEOE_VSA2",	"SMR_VSA7",	"SMR_VSA9"]
Electronicas = train_data_scaled_electronic[propiedades_electronicas_a_usar]
columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas, Electronicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
#xgboost_MACCS_ECFP_fisicoquimicas_electronicas, cm, y_prob_ECFP_MACCS_FISICOQUIMICOS_electronicos, y_test_ECFP_MACCS_FISICOQUIMICOS_electronicos, X_train, X_test = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS +  Fisicoquimicas')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
train_data_scaled_electronic_test = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_scaled_electronic_test.csv')
Electronicas_TEST = train_data_scaled_electronic_test[propiedades_electronicas_a_usar]
columns_subset_test = pd.concat([ECFP_test, MACCS_test, Fisicoquimicas_test, Electronicas_TEST], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)

predict_data = columns_subset_test
#nuevas_predicciones = xgboost_MACCS_ECFP_fisicoquimicas_electronicas.predict(predict_data)
#test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
#y_true = test_data_fingerprints['Clasificacion_ATS']
#y_pred = test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS']
# Matriz de confusión
#cm = confusion_matrix(y_true, y_pred)
#print("Matriz de Confusión:")
#print(cm)
#Cálculo de métricas
#precision = precision_score(y_true, y_pred)
#recall = recall_score(y_true, y_pred)
#f1 = f1_score(y_true, y_pred)
#accuracy = accuracy_score(y_true, y_pred)

# Mostrar las métricas
#print(f"Precisión: {precision}")
#print(f"Recall: {recall}")
#print(f"F1-Score: {f1}")
#print(f"Exactitud: {accuracy}")

from modules.procesamiento.graficas import graficar_roc_multiple
#modelos = [xgboost_MACCS_ECFP_fisicoquimicas , xgboost_MACCS_ECFP_fisicoquimicas_electronicas ]
#X_test_list = [y_prob_ECFP_MACCS_FISICOQUIMICOS, y_prob_ECFP_MACCS_FISICOQUIMICOS_electronicos]
#y_test_list = [y_test_ECFP_MACCS_FISICOQUIMICOS, y_test_ECFP_MACCS_FISICOQUIMICOS_electronicos]
#etiquetas_modelos = ["Modelo_ECFP_MACCS_FISICOQUIMICOS_XGBOOST_BALANCED", "Modelo_ECFP_MACCS_FISICOQUIMICOS_XGBOOST_BALANCED_electronicas"]
#graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos)


"""
SHAP MODELO DE ENTRENAMIENTO
"""
#explainer = shap.TreeExplainer(xgboost_MACCS_ECFP_fisicoquimicas_electronicas, X_train)
#shap_values = explainer(X_test)  # Sin la columna de predicción
# Crear el gráfico SHAP con título

# Crear la figura y agregar título
#plt.figure()
#shap.summary_plot(shap_values, X_test, show=False)  # Desactiva el show automático de SHAP
#plt.title("SHAP Modelo dataset desbalanceado + área superficial", fontsize=14, pad=30)  # Aumenta el pad
# Ajustar la posición del gráfico
#plt.subplots_adjust(top=0.85)  # Reduce este valor para bajar más el gráfico
# Guardar la figura
#plt.savefig('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/graficas/shap_modelo_areasuperf.png', dpi=300)
#plt.show()  # Mostrar la figura en pantalla


"""
EXTRAER INCI NAME, IUPAC NAME, CAS DE LAS SUSTANCIAS PREDICHAS
"""
predicciones_COSING = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/predicciones_COSING.csv')
PFAS_cosing_SMILES = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/PFAS_cosing_SMILES.csv')
import pandas as pd

# Realizar el join asegurando que predicciones_COSING mantiene su tamaño
PFAS_COSING_INFORMATION = predicciones_COSING.merge(PFAS_cosing_SMILES, on="SMILES", how="left")
print(PFAS_COSING_INFORMATION. columns)
# Verificar el tamaño del resultado
#guardar_csv(PFAS_COSING_INFORMATION, 'C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/PFAS_COSING_INFORMATION.csv')


Analisis_resultados_Predicciones = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/predicciones_COSING.csv')

"""
Histograma De los resultados predichos
"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
ax = sns.countplot(x=Analisis_resultados_Predicciones["Predicción"], palette=["#8da0cb", "#fc8d62"])

# Etiquetas con los valores exactos
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.xlabel("Clase Predicha")
plt.ylabel("Frecuencia")
plt.title("Distribución de Predicciones")
plt.xticks([0, 1], ["Inactivo", "Activo"])  # Etiquetas personalizadas
plt.grid(axis='y', linestyle="--", alpha=0.7)

#plt.show()



import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,4))
sns.boxplot(data=Analisis_resultados_Predicciones[["Probabilidad_Act", "Probabilidad_Inac"]])
plt.xlabel("Clase")
plt.ylabel("Probabilidad")
plt.title("Boxplot de Probabilidades por Clase")
plt.xticks([0, 1], ["Activo", "Inactivo"])
#plt.show()

"""
Hisotgramas para las probabilidades act e inac
"""
plt.figure(figsize=(10, 4))

# Histograma para Probabilidad_Act
plt.subplot(1, 2, 1)  # Primera figura (izquierda)
sns.histplot(Analisis_resultados_Predicciones["Probabilidad_Act"], bins=20, color="#4c72b0", kde=True)
plt.xlabel("Probabilidad de ser Activo")
plt.ylabel("Frecuencia")
plt.title("Distribución de Probabilidad Activo")

# Histograma para Probabilidad_Inac
plt.subplot(1, 2, 2)  # Segunda figura (derecha)
sns.histplot(Analisis_resultados_Predicciones["Probabilidad_Inac"], bins=20, color="#dd8452", kde=True)
plt.xlabel("Probabilidad de ser Inactivo")
plt.ylabel("Frecuencia")
plt.title("Distribución de Probabilidad Inactivo")

plt.tight_layout()  # Ajustar los gráficos para que no se solapen
#plt.show()

"""
VISUALIZAR TODOS LOS MACCS
"""

from rdkit import Chem
from rdkit.Chem import MACCSkeys

# Obtener todas las definiciones de los bits MACCS
maccs_definitions = MACCSkeys.smartsPatts
#print("Claves MACCS")
#print(maccs_definitions)

# Ver las primeras claves para entender la indexación
#print(list(maccs_definitions.keys())[:10])  # Muestra los primeros 10 índices

# Revisar si el bit 112 está en índice 112
bit_112 = maccs_definitions[112] if 112 in maccs_definitions else "No definido"
print(f"El bit 112 en MACCS (índice 112 en Python) representa: {bit_112}")

# Revisar si los bits están en el diccionario de definiciones de MACCS
bits_interes = [154, 99, 49, 44, 98, 109, 72, 166, 145, 129]

for bit in bits_interes:
    bit_definicion = maccs_definitions[bit] if bit in maccs_definitions else "No definido"
    print(f"El bit {bit} en MACCS (índice {bit} en Python) representa: {bit_definicion}")

"""
visualizar las estructuras
"""

from rdkit import Chem
from rdkit.Chem import Draw

# Subestructura del bit 112 en MACCS
smarts_pattern = "[!#6;!#1]1~*~*~*~*~*~1"  # Representación SMARTS del bit 112
substructure_mol = Chem.MolFromSmarts(smarts_pattern)

# Verifica si la conversión fue exitosa
if substructure_mol:
    # Guardar la imagen en un archivo
    img_path = "C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/graficas/MACCS_98.png"
    Draw.MolToFile(substructure_mol, img_path, size=(300, 300))
    print(f"Imagen guardada en {img_path}")
else:
    print("Error al generar la subestructura.")

"""
Grafico de dispersión MACCS_154.1 + PREDICION + PESO MOLECULAR
"""

plt.figure(figsize=(8, 6))
sns.violinplot(
    data=Analisis_resultados_Predicciones,
    x="154.1",
    y="Peso_Molecular_scaled",
    hue="Predicción",
    split=True,
    palette={0: "blue", 1: "red"}
)
plt.xlabel("MACCS_154.1")
plt.ylabel("Peso Molecular escalado")
plt.title("Densidad del Peso Molecular según MACCS_154.1 y Predicción")
plt.legend(title="Predicción (0=Inactivo, 1=Activo)")
#plt.show()

"""
Graficas de MACCS 154
"""

from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors


def highlight_substructures(df, smiles_col, bit_idx):
    mols = [Chem.MolFromSmiles(smiles) for smiles in df[smiles_col] if Chem.MolFromSmiles(smiles) is not None]
    fps = [rdMolDescriptors.GetMACCSKeysFingerprint(mol) for mol in mols]

    selected_mols = [mol for mol, fp in zip(mols, fps) if fp[bit_idx]]

    img = Draw.MolsToGridImage(selected_mols, molsPerRow=4, subImgSize=(300, 300),
                               legends=["Molecule {}".format(i + 1) for i in range(len(selected_mols))])
    img.save("highlighted_substructures.png")


# Llamar la función con el DataFrame y la columna correspondiente
highlight_substructures(Analisis_resultados_Predicciones, "SMILES", 154)

"""
Resaltado de BIT 154
"""

import pandas as pd
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from io import BytesIO

# Filtrar moléculas donde MACCS_154.1 == 1
subset_df = Analisis_resultados_Predicciones[Analisis_resultados_Predicciones['154.1'] == 1]

# Convertir SMILES a moléculas RDKit
molecules = [Chem.MolFromSmiles(smiles) for smiles in subset_df['SMILES']]

# Definir la subestructura de MACCS_154.1
substructure = Chem.MolFromSmarts("[#6]=[#6]")  # SMARTS correspondiente

# Crear imágenes resaltando la subestructura con un círculo
highlighted_images = []
for mol in molecules:
    if mol:
        atom_indices = mol.GetSubstructMatches(substructure)
        highlight_atoms = [idx for match in atom_indices for idx in match] if atom_indices else []

        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)

        # Personalizar opciones de resaltado (círculos)
        drawer.SetDrawOptions(drawer.drawOptions())
        drawer.drawOptions().circleHighlight = True  # Activa círculos en los átomos resaltados

        drawer.FinishDrawing()
        # Convertir la imagen binaria en formato PIL
        img_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(img_data))
        highlighted_images.append(img)

# Guardar la primera imagen de ejemplo (puedes adaptarlo para guardarlas todas)
highlighted_images[40].save("molecula_resaltada_40.png")
highlighted_images[41].save("molecula_resaltada_41.png")
highlighted_images[42].save("molecula_resaltada_42.png")

#print(subset_df)

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from io import BytesIO

# Definir la subestructura de MACCS_154.1
substructure = Chem.MolFromSmarts("[#6]=[#8]")  # SMARTS correspondiente


# Función para dibujar la molécula resaltando la subestructura
def dibujar_molecula(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        atom_indices = mol.GetSubstructMatches(substructure)
        highlight_atoms = [idx for match in atom_indices for idx in match] if atom_indices else []

        # Crear el drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.drawOptions().circleAtoms = True  # Activa los círculos en los átomos resaltados
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        drawer.FinishDrawing()

        # Convertir la imagen binaria en formato PIL
        img_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(img_data))

        # Mostrar la imagen
        img.show()

        return img
    else:
        print("Error: SMILES inválido")


# Ingresar un SMILES manualmente
smiles_input = "C(=O)(C(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O"
#img = dibujar_molecula(smiles_input)

# Guardar la imagen si se desea
if img:
    img.save("molecula_resaltada_fluoro.png")

"""
Analisis del MACCS_99.1
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

# Crear una copia del dataframe con la columna correcta
df_plot = Analisis_resultados_Predicciones.copy()

# Reemplazar valores en la columna 'Predicción'
df_plot['Predicción'] = df_plot['Predicción'].map({0: 'Inactivo', 1: 'Activo'})
df_plot['145.1'] = df_plot['145.1'].map({0: 'Ausente_145', 1: 'Presente_145'})
df_plot['99.1'] = df_plot['99.1'].map({0: 'Ausente_99', 1: 'Presente_99'})

# Definir colores personalizados
color_palette = {'Inactivo': '#8da0cb', 'Activo': '#fd8d83'}

# Crear el gráfico de mosaico
plt.figure(figsize=(10, 6))
mosaic(df_plot, ['145.1', '99.1', 'Predicción'], title='Relación entre 145.1, 99.1 y Predicción', properties=lambda key: {'color': color_palette[key[2]]})
#plt.show()


"""
Graficar las moleculas con dos bits a la vez
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from io import BytesIO

# Filtrar moléculas donde MACCS_154.1 == 1 o MACCS_99.1 == 1
subset_df = Analisis_resultados_Predicciones[
    (Analisis_resultados_Predicciones['145.1'] == 1) & (Analisis_resultados_Predicciones['99.1'] == 1)
    ]

# Convertir SMILES a moléculas RDKit
molecules = [Chem.MolFromSmiles(smiles) for smiles in subset_df['SMILES']]

# Definir las subestructuras de MACCS_154.1 y MACCS_99.1
substructure_145 = Chem.MolFromSmarts("*1~*~*~*~*~*~1")  # Ejemplo de SMARTS
substructure_99 = Chem.MolFromSmarts("[#6]=[#6]")  # Otro ejemplo de SMARTS

# Asignar colores a cada bit
highlight_colors = {154: (1.0, 0.0, 0.0),  # Rojo para MACCS_154.1
                    99: (0.0, 0.0, 1.0)}  # Azul para MACCS_99.1

# Crear imágenes resaltando ambas subestructuras
highlighted_images = []
for mol in molecules:
    if mol:
        # Encontrar los átomos que coinciden con las subestructuras
        atom_indices_145 = [idx for match in mol.GetSubstructMatches(substructure_145) for idx in
                            match] if substructure_145 else []
        atom_indices_99 = [idx for match in mol.GetSubstructMatches(substructure_99) for idx in
                           match] if substructure_99 else []

        # Solo continuar si la molécula contiene ambas subestructuras
        if atom_indices_145 and atom_indices_99:
            highlight_atoms = atom_indices_145 + atom_indices_99
            highlight_atom_colors = {}

        for idx in atom_indices_145:
            highlight_atom_colors[idx] = highlight_colors[154]
        for idx in atom_indices_99:
            highlight_atom_colors[idx] = highlight_colors[99]

        # Dibujar la molécula
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors)
        drawer.FinishDrawing()

        # Convertir la imagen binaria en formato PIL
        img_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(img_data))
        highlighted_images.append(img)

highlighted_images[0].save("molecula_resaltada_0.png")
highlighted_images[1].save("molecula_resaltada_1.png")
highlighted_images[2].save("molecula_resaltada_2.png")
highlighted_images[4].save("molecula_resaltada_4.png")

print(subset_df['SMILES'])

"""
ANALISIS 49.1 MACCS
"""
import seaborn as sns
import matplotlib.pyplot as plt
df_grafico = Analisis_resultados_Predicciones.copy()
df_grafico['Predicción'] = df_grafico['Predicción'].map({0: 'Inactivo', 1: 'Activo'})
df_grafico['49.1'] = df_grafico['49.1'].map({0: 'Ausente', 1: 'Presente'})
# Crear el gráfico de cajas
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_grafico,
    x="49.1",
    y="LogP_scaled",
    hue="Predicción"
    )

# Añadir títulos y etiquetas
plt.title("Distribución de Log_P_scaled según MACCS_49.1 y Predicción")
plt.xlabel("MACCS_49.1")
plt.ylabel("LogP_scaled")
plt.legend(title="Predicción")

# Mostrar el gráfico
#plt.show()

"""
Resaltado de BIT 49
"""

import pandas as pd
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from io import BytesIO

# Filtrar moléculas donde MACCS_154.1 == 1
subset_df = Analisis_resultados_Predicciones[Analisis_resultados_Predicciones['49.1'] == 1]

# Convertir SMILES a moléculas RDKit
molecules = [Chem.MolFromSmiles(smiles) for smiles in subset_df['SMILES']]

# Definir la subestructura de MACCS_154.1
substructure = Chem.MolFromSmarts("[!+0]")  # SMARTS correspondiente

# Crear imágenes resaltando la subestructura con un círculo
highlighted_images = []
for mol in molecules:
    if mol:
        atom_indices = mol.GetSubstructMatches(substructure)
        highlight_atoms = [idx for match in atom_indices for idx in match] if atom_indices else []

        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)

        # Personalizar opciones de resaltado (círculos)
        drawer.SetDrawOptions(drawer.drawOptions())
        drawer.drawOptions().circleHighlight = True  # Activa círculos en los átomos resaltados

        drawer.FinishDrawing()
        # Convertir la imagen binaria en formato PIL
        img_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(img_data))
        highlighted_images.append(img)

# Guardar la primera imagen de ejemplo (puedes adaptarlo para guardarlas todas)
highlighted_images[0].save("molecula_resaltada_49.0.png")
highlighted_images[1].save("molecula_resaltada_49.1.png")
highlighted_images[2].save("molecula_resaltada_49.2.png")
highlighted_images[3].save("molecula_resaltada_49.3.png")

"""
ANALISIS 44.1
"""
df_grafico = Analisis_resultados_Predicciones.copy()
df_grafico['Predicción'] = df_grafico['Predicción'].map({0: 'Inactivo', 1: 'Activo'})
df_grafico['44.1'] = df_grafico['44.1'].map({0: 'Ausente', 1: 'Presente'})
# Crear el gráfico de cajas
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_grafico,
    x="44.1",
    y="LogP_scaled",
    hue="Predicción"
    )

# Añadir títulos y etiquetas
plt.title("Distribución de Log_P_scaled según MACCS_44.1 y Predicción")
plt.xlabel("MACCS_44.1")
plt.ylabel("LogP_scaled")
plt.legend(title="Predicción")

# Mostrar el gráfico
plt.show()

"""
Visualizacion moleculas 44.1
"""

# Ingresar un SMILES manualmente
smiles_input = "F[Sn]F"
img = dibujar_molecula(smiles_input)

smiles_input = "C(CP(=O)(O)[O-])C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F.[Na+]"
img = dibujar_molecula(smiles_input)


"""
ANALISIS ECFP 1873
"""

df_grafico = Analisis_resultados_Predicciones.copy()
df_grafico['Predicción'] = df_grafico['Predicción'].map({0: 'Inactivo', 1: 'Activo'})
df_grafico['1873'] = df_grafico['1873'].map({0: 'Ausente', 1: 'Presente'})
# Crear el gráfico de cajas
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_grafico,
    x="1873",
    y="LogP_scaled",
    hue="Predicción"
    )

# Añadir títulos y etiquetas
plt.title("Distribución de Log_P_scaled según ECFP_1873 y Predicción")
plt.xlabel("ECFP_1873")
plt.ylabel("LogP_scaled")
plt.legend(title="Predicción")

# Mostrar el gráfico
plt.show()


"""
VISUALIZAR 1873 
"""
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from rdkit.Chem import Draw

from rdkit.Chem import Draw
from rdkit import Chem
mol = AllChem.MolFromSmiles('C1CN(CCC1(C2=CC(=CC=C2)C(F)(F)F)O)CCCC(=O)C3=CC=C(C=C3)F')
#fingerprint = df['ECFP'][0]
# Generar bitInfo para almacenar la información de los bits activados
bit_info = {}
fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, bitInfo=bit_info)
list_bits = [(mol, bit, bit_info) for bit in fingerprint.GetOnBits()]
legends = [str(bit) for bit in fingerprint.GetOnBits()]


img = Draw.DrawMorganBits(list_bits, molsPerRow=4,legends=legends)

"""
CODIGO PARA VISUALIZAR LOS FINGERPRINTS
"""
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image

# Suponiendo que 'img' es el objeto SVG que generaste
svg_file_path = 'output_1.svg'  # La ruta donde guardas el SVG
png_file_path = 'output_1.png'  # La ruta donde guardas el PNG

# Guarda la imagen SVG
with open(svg_file_path, 'w') as f:
    f.write(str(img))

# Cargar el archivo SVG y convertirlo a un gráfico que pueda manejar ReportLab
drawing = svg2rlg(svg_file_path)

"""
Analisis sustancias perfluoradas
"""

perfluoradas = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/perfluoradas.csv')
#borrar columnas repetidas
perfluoradas = perfluoradas.drop_duplicates(subset=['INCI Name/Substance Name'], keep='first')

#perfluoradas_INFO = perfluoradas[['INCI Name/Substance Name', 'Predicción', 'Probabilidad_Inac', 'Probabilidad_Act', 'Annex/Ref']]
#guardar_csv(perfluoradas_INFO, 'C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/perfluoradas_INFO.csv')

perfluoradas_SHAP = perfluoradas.drop(columns=['INCI Name/Substance Name',	'Predicción',	'Probabilidad_Inac',
                                          'Probabilidad_Act','Annex/Ref', 'IUPAC',	'Type', 'CAS No.', 'SMILES'])


"""
SHAP MODELO PREDICCIONES
"""
explainer = shap.TreeExplainer(xgboost_MACCS_ECFP_fisicoquimicas)
shap_values = explainer(perfluoradas_SHAP)  # Sin la columna de predicción
# Crear el gráfico SHAP con título

# Crear la figura y agregar título
plt.figure()
shap.summary_plot(shap_values, perfluoradas_SHAP, show=False)  # Desactiva el show automático de SHAP
plt.title("SHAP Perfluoradas Predicciones", fontsize=14, pad=30)  # Aumenta el pad
# Ajustar la posición del gráfico
plt.subplots_adjust(top=0.85)  # Reduce este valor para bajar más el gráfico
# Guardar la figura
plt.savefig('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/graficas/shap_modelo_Predicciones_perfluoradas.png', dpi=300)
plt.show()  # Mostrar la figura en pantalla

"""
VISUALIZAR 1873 
"""
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from rdkit.Chem import Draw

from rdkit.Chem import Draw
from rdkit import Chem
mol = AllChem.MolFromSmiles('C(CP(=O)(O)[O-])C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F.[Na+]')
#fingerprint = df['ECFP'][0]
# Generar bitInfo para almacenar la información de los bits activados
bit_info = {}
fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, bitInfo=bit_info)
list_bits = [(mol, bit, bit_info) for bit in fingerprint.GetOnBits()]
legends = [str(bit) for bit in fingerprint.GetOnBits()]


img = Draw.DrawMorganBits(list_bits, molsPerRow=4,legends=legends)

"""
CODIGO PARA VISUALIZAR LOS FINGERPRINTS
"""
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image

# Suponiendo que 'img' es el objeto SVG que generaste
svg_file_path = 'output_3.svg'  # La ruta donde guardas el SVG
png_file_path = 'output_3.png'  # La ruta donde guardas el PNG

# Guarda la imagen SVG
with open(svg_file_path, 'w') as f:
    f.write(str(img))

# Cargar el archivo SVG y convertirlo a un gráfico que pueda manejar ReportLab
drawing = svg2rlg(svg_file_path)





"""
MOLECULAS CON 1873 PRESENTE
"""
perfluoradas_SHAP_1873 = perfluoradas_SHAP[perfluoradas_SHAP['1873'] == 1]
print(perfluoradas_SHAP_1873)
perfluoradas_SHAP_1398 = perfluoradas_SHAP[perfluoradas_SHAP['1398'] == 0]
print(perfluoradas_SHAP_1398)
perfluoradas_SHAP_99 = perfluoradas_SHAP[perfluoradas_SHAP['99.1'] == 1]
print(perfluoradas_SHAP_99)

"""
Resaltado de BIT 109
"""

import pandas as pd
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from io import BytesIO

# Filtrar moléculas donde MACCS_154.1 == 1
subset_df = Analisis_resultados_Predicciones[Analisis_resultados_Predicciones['109.1'] == 1]

# Convertir SMILES a moléculas RDKit
molecules = [Chem.MolFromSmiles(smiles) for smiles in subset_df['SMILES']]

# Definir la subestructura de MACCS_154.1
substructure = Chem.MolFromSmarts("*~[CH2]~[#8]")  # SMARTS correspondiente

# Crear imágenes resaltando la subestructura con un círculo
highlighted_images = []
for mol in molecules:
    if mol:
        atom_indices = mol.GetSubstructMatches(substructure)
        highlight_atoms = [idx for match in atom_indices for idx in match] if atom_indices else []

        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)

        # Personalizar opciones de resaltado (círculos)
        drawer.SetDrawOptions(drawer.drawOptions())
        drawer.drawOptions().circleHighlight = True  # Activa círculos en los átomos resaltados

        drawer.FinishDrawing()
        # Convertir la imagen binaria en formato PIL
        img_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(img_data))
        highlighted_images.append(img)

# Guardar la primera imagen de ejemplo (puedes adaptarlo para guardarlas todas)
highlighted_images[0].save("molecula_resaltada_109.0.png")
highlighted_images[1].save("molecula_resaltada_109.1.png")
highlighted_images[2].save("molecula_resaltada_109.2.png")
highlighted_images[3].save("molecula_resaltada_109.3.png")

"""
ANALISIS 109.1
"""
df_grafico = Analisis_resultados_Predicciones.copy()
df_grafico['Predicción'] = df_grafico['Predicción'].map({0: 'Inactivo', 1: 'Activo'})
df_grafico['109.1'] = df_grafico['109.1'].map({0: 'Ausente', 1: 'Presente'})
# Crear el gráfico de cajas
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_grafico,
    x="109.1",
    y="NumRotatableBonds_scaled",
    hue="Predicción"
    )

# Añadir títulos y etiquetas
plt.title("Distribución de NumRotatableBonds_scaled según MACCS_109.1 y Predicción")
plt.xlabel("MACCS_109.1")
plt.ylabel("NumRotatableBonds_scaled")
plt.legend(title="Predicción")

# Mostrar el gráfico
plt.show()

"""
Resaltado de BIT 98
"""

import pandas as pd
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from io import BytesIO

# Filtrar moléculas donde MACCS_154.1 == 1
subset_df = Analisis_resultados_Predicciones[Analisis_resultados_Predicciones['98.1'] == 1]

# Convertir SMILES a moléculas RDKit
molecules = [Chem.MolFromSmiles(smiles) for smiles in subset_df['SMILES']]

# Definir la subestructura de MACCS_154.1
substructure = Chem.MolFromSmarts("[!#6;!#1]1~*~*~*~*~*~1")  # SMARTS correspondiente

# Crear imágenes resaltando la subestructura con un círculo
highlighted_images = []
for mol in molecules:
    if mol:
        atom_indices = mol.GetSubstructMatches(substructure)
        highlight_atoms = [idx for match in atom_indices for idx in match] if atom_indices else []

        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)

        # Personalizar opciones de resaltado (círculos)
        drawer.SetDrawOptions(drawer.drawOptions())
        drawer.drawOptions().circleHighlight = True  # Activa círculos en los átomos resaltados

        drawer.FinishDrawing()
        # Convertir la imagen binaria en formato PIL
        img_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(img_data))
        highlighted_images.append(img)

# Guardar la primera imagen de ejemplo (puedes adaptarlo para guardarlas todas)
highlighted_images[0].save("molecula_resaltada_98.0.png")
highlighted_images[1].save("molecula_resaltada_98.1.png")
highlighted_images[2].save("molecula_resaltada_98.2.png")
highlighted_images[3].save("molecula_resaltada_98.3.png")

"""
ANALISIS 98.1
"""
df_grafico = Analisis_resultados_Predicciones.copy()
df_grafico['Predicción'] = df_grafico['Predicción'].map({0: 'Inactivo', 1: 'Activo'})
df_grafico['98.1'] = df_grafico['98.1'].map({0: 'Ausente', 1: 'Presente'})
# Crear el gráfico de cajas
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_grafico,
    x="98.1",
    y="129.1",
    hue="Predicción"
    )

# Añadir títulos y etiquetas
plt.title("Distribución de Log_P_scaled según MACCS_109.1 y Predicción")
plt.xlabel("MACCS_98.1")
plt.ylabel("MACCS_129.1")
plt.legend(title="Predicción")

# Mostrar el gráfico
plt.show()

"""
Ver el bit 129 subestructuralmente
"""
from rdkit import Chem
from rdkit.Chem import Draw

# Definimos los SMARTS individuales extraídos del bit 129
smarts_list = [
    "*~[CH2]~*~*~[CH2]~*",
    "[R]1@[CH2]@[R]@[R]@[CH2;R]1",
    "*~[CH2]~[R]1@[R]@[CH2;R]1"
]

# Convertir cada SMARTS en una molécula
mols = [Chem.MolFromSmarts(s) for s in smarts_list]
# Verificar si la conversión fue exitosa
mols = [mol for mol in mols if mol is not None]

if mols:
    # Guardar la imagen en un archivo con una cuadrícula de los fragmentos
    img_path = "C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/graficas/MACCS_129.png"
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 300))
    img.save(img_path)
    print(f"Imagen guardada en {img_path}")
else:
    print("Error al generar las subestructuras.")