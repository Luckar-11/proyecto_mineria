import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Iniciando Misión 1: Entrenamiento del Modelo...")

# 1. Cargar los Datos
try:
    df = pd.read_csv("machine failure.csv")
    print("Datos cargados correctamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'machine failure.csv'.")
    exit()

# 2. Limpieza y Preparación de Datos
# Renombramos columnas para que sean más fáciles de usar
df = df.rename(columns={
    'Air temperature [K]': 'temp_aire',
    'Process temperature [K]': 'temp_proceso',
    'Rotational speed [rpm]': 'velocidad_rotacion',
    'Torque [Nm]': 'torque',
    'Tool wear [min]': 'desgaste_herramienta'
})

# Quitamos columnas que no ayudan a predecir o que "hacen trampa"
# (no sabremos el tipo de falla (TWF, HDF...) antes de que ocurra)
df = df.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

# Convertimos la columna 'Type' (L, M, H) en números
df_dummies = pd.get_dummies(df, columns=['Type'], drop_first=True)
print("Datos limpiados y preparados.")

# 3. Definir nuestro objetivo (X) y lo que queremos predecir (y)
X = df_dummies.drop(columns=['Machine failure'])  # Todas las columnas MENOS la respuesta
y = df_dummies['Machine failure']                # Solo la columna de respuesta (0 o 1)

# Guardamos los nombres de las columnas en el orden exacto que el modelo espera
columnas_del_modelo = X.columns.tolist()
joblib.dump(columnas_del_modelo, 'columnas_modelo.pkl')
print(f"Modelo entrenado con estas columnas: {columnas_del_modelo}")

# 4. Dividir los datos: 80% para entrenar, 20% para probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Elegir y Entrenar el Modelo (Random Forest)
# class_weight='balanced' es CLAVE porque tienes muy pocas fallas (339 vs 9661)
modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
print("Entrenando el modelo... (esto puede tomar unos segundos)")
modelo.fit(X_train, y_train)
print("¡Modelo entrenado!")

# 6. Evaluar el Modelo (¿Qué tan bueno es?)
y_pred = modelo.predict(X_test)
print("\n--- Evaluación del Modelo ---")
# Fíjate en la fila "1" (falla), especialmente en 'recall'.
# Nos dice cuántas fallas reales logramos "atrapar".
print(classification_report(y_test, y_pred))

# 7. ¡Guardar el Modelo!
# Este es el archivo que usará FastAPI
joblib.dump(modelo, 'modelo_fallas.pkl')
print("\n¡Misión 1 Completa! Modelo guardado en 'modelo_fallas.pkl'")