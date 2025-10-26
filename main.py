from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sqlalchemy.orm import Session
from typing import Optional

# Importar la configuración de la BD (de los archivos que creamos antes)
# CAMBIO: Se quitaron los puntos (.) de las importaciones
import models
import schemas
from database import SessionLocal, engine

# Crear las tablas en la BD (si no existen)
models.Base.metadata.create_all(bind=engine)

# 1. Inicializar la aplicación FastAPI
app = FastAPI(title="API de Predicción de Fallas de Maquinaria v2.1 (con BD)")

# --- Dependencia de Sesión de BD ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 2. Cargar TODOS los modelos y helpers
try:
    modelo_falla = joblib.load("modelo_fallas.pkl")
    columnas_modelo = joblib.load("columnas_modelo.pkl")
    modelo_tipo_falla = joblib.load("modelo_tipo_falla.pkl")
    labels_tipo_falla = joblib.load("labels_tipo_falla.pkl")
    print("Todos los modelos cargados exitosamente. ¡Listos para predecir!")
except FileNotFoundError:
    print("Error: Faltan archivos .pkl. Asegúrate de ejecutar 'entrenar.py' primero.")
    modelo_falla = None
    modelo_tipo_falla = None

# 3. Diccionario de Recomendaciones
RECOMENDACIONES = {
    "TWF": "Falla por Desgaste de Herramienta (TWF). Revisar la herramienta de corte, posible reemplazo necesario.",
    "HDF": "Falla por Disipación de Calor (HDF). Inspeccionar el sistema de refrigeración y las temperaturas de proceso.",
    "PWF": "Falla de Potencia (PWF). Verificar la fuente de alimentación y posibles picos de torque.",
    "OSF": "Falla por Sobreesfuerzo (OSF). Reducir la velocidad de rotación o el torque aplicado.",
    "OTRA": "Falla Indeterminada. Realizar una inspección general de la máquina."
}

# 4. Definir la estructura de los datos que esperamos recibir
#    ¡HEMOS AÑADIDO machine_id PARA SABER QUÉ MÁQUINA ES!
class DatosMaquina(BaseModel):
    machine_id: int  # <-- ¡NUEVO!
    temp_aire: float
    temp_proceso: float
    velocidad_rotacion: int
    torque: float
    desgaste_herramienta: int
    Type: str  # "L", "M", o "H"

# 5. Endpoint de bienvenida
@app.get("/")
def bienvenida():
    return {"mensaje": "API del Doctor de Máquinas v2.1 está funcionando."}

# 6. Endpoint de predicción (Actualizado con "db: Session")
@app.post("/predecir")
def predecir_falla(datos: DatosMaquina, db: Session = Depends(get_db)):
    if not modelo_falla or not modelo_tipo_falla:
        raise HTTPException(status_code=500, detail="Modelos no cargados. Revisa la consola del backend.")

    # --- VERIFICACIÓN DE MÁQUINA (NUEVO) ---
    # Es buena práctica verificar que la máquina existe en la tabla 'machines'
    db_machine = db.query(models.Machine).filter(models.Machine.machine_id == datos.machine_id).first()
    if db_machine is None:
        raise HTTPException(status_code=404, detail=f"Máquina con ID {datos.machine_id} no encontrada.")

    # 7. Procesar los datos de entrada (Tu lógica original)
    # Excluimos machine_id del dataframe para el modelo
    input_data_dict = datos.model_dump(exclude={'machine_id'})
    input_df = pd.DataFrame([input_data_dict])
    input_dummies = pd.get_dummies(input_df, columns=['Type'])
    
    input_final = pd.DataFrame(columns=columnas_modelo)
    input_final = pd.concat([input_final, input_dummies]).fillna(0)[columnas_modelo]

    try:
        # --- PREDICCIÓN MODELO 1 ---
        prediccion_falla = modelo_falla.predict(input_final)
        probabilidad_falla = modelo_falla.predict_proba(input_final)
        
        resultado_falla = int(prediccion_falla[0])
        confianza = float(probabilidad_falla[0][resultado_falla]) * 100

        # Declarar variables para guardar en BD
        tipo_falla_str = "N/A"
        recomendacion_str = "Continuar operación estándar."
        falla_bool = (resultado_falla == 1)
        tipos_falla_obj = {} # Para la tabla 'failure_types'

        # 8. Decidir la respuesta
        if resultado_falla == 1:
            # --- CASO: FALLA PROBABLE ---
            
            # --- PREDICCIÓN MODELO 2 ---
            prediccion_tipo = modelo_tipo_falla.predict(input_final)
            
            falla_str_lista = []
            rec_str_lista = []
            
            for i, label in enumerate(labels_tipo_falla):
                if prediccion_tipo[0][i] == 1:
                    recomendacion = RECOMENDACIONES.get(label, "Revisión requerida.")
                    falla_str_lista.append(recomendacion.split('.')[0])
                    rec_str_lista.append(recomendacion.split('.')[1].strip())
                    tipos_falla_obj[label] = True # Guardar para la BD
                else:
                    tipos_falla_obj[label] = False # Guardar para la BD

            if not falla_str_lista:
                tipo_falla_str = RECOMENDACIONES["OTRA"].split('.')[0]
                recomendacion_str = RECOMENDACIONES["OTRA"].split('.')[1].strip()
                tipos_falla_obj["RNF"] = True # Marcar Falla Aleatoria (Random)
            else:
                tipo_falla_str = ", ".join(falla_str_lista)
                recomendacion_str = " ".join(rec_str_lista)

        # --- SECCIÓN DE BASE DE DATOS (NUEVO) ---
        
        # 9. Guardar la lectura en la tabla 'machine_readings'
        db_reading = models.MachineReading(
            machine_id = datos.machine_id,
            air_temperature = datos.temp_aire,
            process_temperature = datos.temp_proceso,
            rotational_speed = datos.velocidad_rotacion,
            torque = datos.torque,
            tool_wear = datos.desgaste_herramienta,
            machine_failure = falla_bool
            # timestamp se añade por defecto
        )
        db.add(db_reading)
        db.commit()
        db.refresh(db_reading) # ¡Importante! para obtener el 'reading_id' generado

        # 10. Si hubo falla, guardar los detalles en 'failure_types'
        if falla_bool:
            db_failure_details = models.FailureType(
                reading_id = db_reading.reading_id,
                twf = tipos_falla_obj.get("TWF", False),
                hdf = tipos_falla_obj.get("HDF", False),
                pwf = tipos_falla_obj.get("PWF", False),
                osf = tipos_falla_obj.get("OSF", False),
                rnf = tipos_falla_obj.get("RNF", False)
            )
            db.add(db_failure_details)
            db.commit()

        # 11. Devolver la respuesta al usuario
        return {
            "prediccion": "FALLA PROBABLE" if falla_bool else "OPERACION NORMAL",
            "confianza": f"{confianza:.2f}%",
            "tipo_falla_probable": tipo_falla_str,
            "recomendacion": recomendacion_str,
            "reading_saved_id": db_reading.reading_id # Devolver el ID de la lectura guardada
        }

    except Exception as e:
        # Si algo falla (predicción o BD), hacemos rollback
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error durante la predicción o guardado: {str(e)}")

# --- Endpoints de CRUD para Máquinas ---
# (Puedes añadir los endpoints que creamos antes para crear/leer máquinas)

@app.post("/machines/", response_model=schemas.MachineResponse)
def create_machine(machine: schemas.MachineCreate, db: Session = Depends(get_db)):
    # CAMBIO: Usar .model_dump() es más seguro con Pydantic v2
    db_machine = models.Machine(**machine.model_dump())
    db.add(db_machine)
    db.commit()
    db.refresh(db_machine)
    return db_machine

@app.get("/machines/{machine_id}", response_model=schemas.MachineResponse)
def get_machine(machine_id: int, db: Session = Depends(get_db)):
    db_machine = db.query(models.Machine).filter(models.Machine.machine_id == machine_id).first()
    if db_machine is None:
        raise HTTPException(status_code=404, detail="Machine not found")
    return db_machine

