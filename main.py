from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Inicializar la aplicación FastAPI
app = FastAPI(title="API de Predicción de Fallas de Maquinaria")

# 2. Cargar el modelo y las columnas al iniciar
# Esto solo se ejecuta una vez, cuando inicias el servidor
try:
    modelo = joblib.load("modelo_fallas.pkl")
    columnas_modelo = joblib.load("columnas_modelo.pkl")
    print("Modelo y columnas cargados exitosamente. ¡Listos para predecir!")
except FileNotFoundError:
    print("Error: Archivos 'modelo_fallas.pkl' o 'columnas_modelo.pkl' no encontrados.")
    print("Por favor, asegúrate de ejecutar 'entrenar.py' primero.")
    modelo = None
    columnas_modelo = None

# 3. Definir la estructura de los datos que esperamos recibir
# Esto es un "contrato": el frontend DEBE enviarnos estos datos.
# Nota: Usamos los nombres amigables que definimos en el entrenamiento.
class DatosMaquina(BaseModel):
    temp_aire: float
    temp_proceso: float
    velocidad_rotacion: int
    torque: float
    desgaste_herramienta: int
    Type: str  # El usuario nos enviará "L", "M", o "H"

    # Ejemplo de cómo se verán los datos que nos envían:
    # {
    #   "temp_aire": 301.5,
    #   "temp_proceso": 310.8,
    #   "velocidad_rotacion": 1450,
    #   "torque": 45.3,
    #   "desgaste_herramienta": 20,
    #   "Type": "L"
    # }


# 4. Crear el "endpoint" de bienvenida (para probar)
@app.get("/")
def bienvenida():
    return {"mensaje": "API del Doctor de Máquinas está funcionando."}


# 5. Crear el "endpoint" de predicción
# Esta es la URL a la que llamará nuestro frontend
@app.post("/predecir")
def predecir_falla(datos: DatosMaquina):
    if modelo is None or columnas_modelo is None:
        return {"error": "Modelo no cargado. Revisa la consola del backend."}

    # Convertir los datos de entrada (JSON) a un DataFrame de Pandas
    # exactamente como lo hicimos en el entrenamiento.
    input_df = pd.DataFrame([datos.model_dump()]) # Pydantic v2 usa .model_dump()

    # 6. Procesar los datos de entrada IGUAL que en el entrenamiento
    # Convertir 'Type' en dummies (Type_L, Type_M, Type_H)
    input_dummies = pd.get_dummies(input_df, columns=['Type'])
    
    # 7. Alinear las columnas (¡Paso CRÍTICO!)
    # Nos aseguramos de que el df de entrada tenga EXACTAMENTE
    # las mismas columnas, en el mismo orden, que el modelo que entrenamos.
    # Rellena con 0 las columnas que falten (ej. si el input fue 'L', faltarán 'Type_M')
    input_final = pd.DataFrame(columns=columnas_modelo)
    input_final = pd.concat([input_final, input_dummies])
    input_final = input_final.fillna(0)
    
    # Asegurarnos de que el orden sea idéntico
    input_final = input_final[columnas_modelo]

    # 8. Realizar la predicción
    try:
        prediccion = modelo.predict(input_final)
        probabilidad = modelo.predict_proba(input_final)
        
        resultado_prediccion = int(prediccion[0])
        confianza = float(probabilidad[0][resultado_prediccion]) * 100
        
        # 9. Devolver una respuesta clara
        if resultado_prediccion == 1:
            return {
                "prediccion": "FALLA PROBABLE",
                "confianza": f"{confianza:.2f}%"
            }
        else:
            return {
                "prediccion": "OPERACION NORMAL",
                "confianza": f"{confianza:.2f}%"
            }

    except Exception as e:
        return {"error": f"Error durante la predicción: {str(e)}"}