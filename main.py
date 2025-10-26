from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# 1. Inicializar la aplicación FastAPI
app = FastAPI(title="API de Predicción de Fallas de Maquinaria v2.0")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    try:
        # --- PASO A: PRE-PROCESAMIENTO ROBUSTO ---

        # 1. Convertir el JSON de entrada a un DataFrame de una fila
        input_df = pd.DataFrame([datos.model_dump()])

        # 2. Convertir 'Type' en columnas dummies (Type_L, Type_M, Type_H)
        input_dummies = pd.get_dummies(input_df, columns=['Type'])
        
        # 3. Alinear columnas (¡ESTA ES LA MEJORA CRÍTICA!)
        # Usamos .reindex() para forzar que el DataFrame de entrada tenga
        # EXACTAMENTE las mismas columnas (y en el mismo orden) que 'columnas_modelo'.
        # Rellena con 0 las columnas que falten (ej. 'Type_M' si el input fue 'L').
        input_final = input_dummies.reindex(columns=columnas_modelo, fill_value=0)
        
        # Esta línea extra asegura el orden, aunque .reindex ya debería manejarlo
        input_final = input_final[columnas_modelo]

        # --- PASO B: PREDICCIÓN CON MODELO 1 (¿Hay Falla?) ---
        prediccion_falla = modelo_falla.predict(input_final)
        probabilidad_falla = modelo_falla.predict_proba(input_final)
        
        resultado_falla = int(prediccion_falla[0])
        confianza = float(probabilidad_falla[0][resultado_falla]) * 100

        # --- PASO C: DECISIÓN Y RESPUESTA ---
        
        if resultado_falla == 0:
            # --- CASO: OPERACIÓN NORMAL ---
            return {
                "prediccion": "OPERACION NORMAL",
                "confianza": f"{confianza:.2f}%",
                "tipo_falla_probable": "N/A",
                "recomendacion": "Continuar operación estándar."
            }
        
        else:
            # --- CASO: FALLA PROBABLE (Llamar al Modelo 2) ---
            
            # --- PREDICCIÓN MODELO 2 (¿Qué tipo de Falla?) ---
            prediccion_tipo = modelo_tipo_falla.predict(input_final)
            
            falla_str_lista = []
            rec_str_lista = []
            
            for i, label in enumerate(labels_tipo_falla):
                if prediccion_tipo[0][i] == 1:
                    recomendacion = RECOMENDACIONES.get(label, "Revisión requerida.")
                    falla_str_lista.append(recomendacion.split('.')[0]) 
                    rec_str_lista.append(recomendacion.split('.')[1].strip()) 

            if not falla_str_lista:
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

    # Capturar cualquier error inesperado durante la predicción
    except Exception as e:
        return {"error": f"Error interno durante la predicción: {str(e)}"}