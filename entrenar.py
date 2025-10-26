import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Iniciando Misión 1 (Actualizada): Entrenando AMBOS modelos...")

# 1. Cargar los Datos
try:
    df = pd.read_csv("machine failure.csv")
    print("Datos cargados correctamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'machine failure.csv'.")
    exit()

# 2. Limpieza y Preparación de Datos (General)
df_procesado = df.rename(columns={
    'Air temperature [K]': 'temp_aire',
    'Process temperature [K]': 'temp_proceso',
    'Rotational speed [rpm]': 'velocidad_rotacion',
    'Torque [Nm]': 'torque',
    'Tool wear [min]': 'desgaste_herramienta'
})
df_procesado = pd.get_dummies(df_procesado, columns=['Type'], drop_first=True)

# Columnas de features (entradas) que usarán AMBOS modelos
features = [
    'temp_aire', 'temp_proceso', 'velocidad_rotacion', 'torque', 
    'desgaste_herramienta', 'Type_L', 'Type_M'
]
# Asegurarnos de que todas las columnas de features existan
for col in features:
    if col not in df_procesado.columns:
        df_procesado[col] = 0 # En caso de que Type_L o Type_M no estuvieran en el split

print(f"Modelos serán entrenados con estas features: {features}")

# 3. --- ENTRENAMIENTO DEL MODELO 1: ¿HAY FALLA? ---
print("\n--- Entrenando Modelo 1 (Predicción de Falla) ---")
X1 = df_procesado[features]
y1 = df_procesado['Machine failure']

# Guardamos los nombres de las columnas para la API
joblib.dump(features, 'columnas_modelo.pkl')

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42, stratify=y1)

# Le decimos que la clase "1" (falla) pesa 20 VECES MÁS que la clase "0" (normal)
pesos = {0: 1, 1: 20} 
modelo_falla = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=pesos)
modelo_falla.fit(X1_train, y1_train)

# Evaluación Modelo 1
y1_pred = modelo_falla.predict(X1_test)
print("Evaluación Modelo 1 (Falla Sí/No):")
print(classification_report(y1_test, y1_pred))

joblib.dump(modelo_falla, 'modelo_fallas.pkl')
print("¡Modelo 1 ('modelo_fallas.pkl') guardado!")


# 4. --- ENTRENAMIENTO DEL MODELO 2: ¿QUÉ TIPO DE FALLA? ---
print("\n--- Entrenando Modelo 2 (Tipo de Falla) ---")

# Filtramos solo las filas donde SÍ hubo falla
df_solo_fallas = df_procesado[df_procesado['Machine failure'] == 1].copy()

# Definimos las entradas (X2) y las salidas (y2)
# Las entradas son las mismas que antes
X2 = df_solo_fallas[features]
# Las salidas son las columnas de tipo de falla
# RNF (Random No Failure) no nos interesa predecir
labels_tipo_falla = ['TWF', 'HDF', 'PWF', 'OSF']
y2 = df_solo_fallas[labels_tipo_falla]

if len(X2) > 0:
    # Este modelo predice múltiples etiquetas a la vez (ej: [1, 0, 1, 0])
    modelo_tipo_falla = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_tipo_falla.fit(X2, y2)

    joblib.dump(modelo_tipo_falla, 'modelo_tipo_falla.pkl')
    # Guardamos los nombres de las etiquetas que predice
    joblib.dump(labels_tipo_falla, 'labels_tipo_falla.pkl')
    print("¡Modelo 2 ('modelo_tipo_falla.pkl') guardado!")
else:
    print("No se encontraron datos de fallas para entrenar el modelo 2.")

print("\n¡Misión 1 (Actualizada) Completa!")