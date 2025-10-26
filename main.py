from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Inicializar la aplicación FastAPI
app = FastAPI(title="API de Predicción de Fallas de Maquinaria v2.0")

# 2. Cargar TODOS los modelos y helpers
try:
    # Modelo 1: ¿Hay falla?
    modelo_falla = joblib.load("modelo_fallas.pkl")
    columnas_modelo = joblib.load("columnas_modelo.pkl")
    
    # Modelo 2: ¿Qué tipo de falla?
    modelo_tipo_falla = joblib.load("modelo_tipo_falla.pkl")
    labels_tipo_falla = joblib.load("labels_tipo_falla.pkl")
    
    print("Todos los modelos cargados exitosamente. ¡Listos para predecir!")
except FileNotFoundError:
    print("Error: Faltan archivos .pkl. Asegúrate de ejecutar 'entrenar.py' primero.")
    modelo_falla = None
    modelo_tipo_falla = None

# 3. Diccionario de Recomendaciones (¡Aquí pones tu conocimiento!)
RECOMENDACIONES = {
    "TWF": "Falla por Desgaste de Herramienta (TWF). Revisar la herramienta de corte, posible reemplazo necesario.",
    "HDF": "Falla por Disipación de Calor (HDF). Inspeccionar el sistema de refrigeración y las temperaturas de proceso.",
    "PWF": "Falla de Potencia (PWF). Verificar la fuente de alimentación y posibles picos de torque.",
    "OSF": "Falla por Sobreesfuerzo (OSF). Reducir la velocidad de rotación o el torque aplicado.",
    "OTRA": "Falla Indeterminada. Realizar una inspección general de la máquina."
}

# 4. Definir la estructura de los datos que esperamos recibir
class DatosMaquina(BaseModel):
    temp_aire: float
    temp_proceso: float
    velocidad_rotacion: int
    torque: float
    desgaste_herramienta: int
    Type: str  # "L", "M", o "H"

# 5. Endpoint de bienvenida
@app.get("/")
def bienvenida():
    return {"mensaje": "API del Doctor de Máquinas v2.0 está funcionando."}

# 6. Endpoint de predicción (Actualizado)
@app.post("/predecir")
def predecir_falla(datos: DatosMaquina):
    if not modelo_falla or not modelo_tipo_falla:
        return {"error": "Modelos no cargados. Revisa la consola del backend."}

    # 7. Procesar los datos de entrada (IGUAL que en el entrenamiento)
    input_df = pd.DataFrame([datos.model_dump()])
    input_dummies = pd.get_dummies(input_df, columns=['Type'])
    
    # Alinear las columnas (Paso CRÍTICO)
    input_final = pd.DataFrame(columns=columnas_modelo)
    input_final = pd.concat([input_final, input_dummies])
    input_final = input_final.fillna(0)
    input_final = input_final[columnas_modelo]

    try:
        # --- PREDICCIÓN MODELO 1 ---
        prediccion_falla = modelo_falla.predict(input_final)
        probabilidad_falla = modelo_falla.predict_proba(input_final)
        
        resultado_falla = int(prediccion_falla[0])
        confianza = float(probabilidad_falla[0][resultado_falla]) * 100

        # 8. Decidir la respuesta
        if resultado_falla == 0:
            # --- CASO: OPERACIÓN NORMAL ---
            return {
                "prediccion": "OPERACION NORMAL",
                "confianza": f"{confianza:.2f}%",
                "tipo_falla_probable": "N/A",
                "recomendacion": "Continuar operación estándar."
            }
        else:
            # --- CASO: FALLA PROBABLE ---
            
            # --- PREDICCIÓN MODELO 2 ---
            # El modelo 2 devuelve un vector, ej: [[1, 0, 0, 1]] (TWF y OSF)
            prediccion_tipo = modelo_tipo_falla.predict(input_final)
            
            falla_str_lista = []
            rec_str_lista = []
            
            # Iteramos sobre las etiquetas (['TWF', 'HDF', 'PWF', 'OSF'])
            for i, label in enumerate(labels_tipo_falla):
                if prediccion_tipo[0][i] == 1:
                    # Usamos el diccionario de recomendaciones
                    recomendacion = RECOMENDACIONES.get(label, "Revisión requerida.")
                    # El texto "Falla por..." ya está en el diccionario
                    falla_str_lista.append(recomendacion.split('.')[0]) # Tomamos solo la parte del nombre
                    rec_str_lista.append(recomendacion.split('.')[1].strip()) # Tomamos solo la recomendación

            if not falla_str_lista:
                # Si el modelo 1 dijo "falla" pero el 2 no encontró tipo
                tipo_falla_str = RECOMENDACIONES["OTRA"].split('.')[0]
                recomendacion_str = RECOMENDACIONES["OTRA"].split('.')[1].strip()
            else:
                tipo_falla_str = ", ".join(falla_str_lista)
                recomendacion_str = " ".join(rec_str_lista)

            return {
                "prediccion": "FALLA PROBABLE",
                "confianza": f"{confianza:.2f}%",
                "tipo_falla_probable": tipo_falla_str,
                "recomendacion": recomendacion_str
            }

    except Exception as e:
        return {"error": f"Error durante la predicción: {str(e)}"}