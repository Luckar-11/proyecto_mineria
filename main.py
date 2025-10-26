from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware  # Importa el Middleware de CORS
from sqlalchemy.orm import Session
import joblib
import pandas as pd
import warnings

# --- Importaciones de la Base de Datos ---
import models
import schemas  # Importa todos los schemas
from database import engine, get_db  # Importa get_db desde database.py

# --- Importa el router del CRUD ---
import crud_endpoints

# --- Configuración de Advertencias ---
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Creación de Tablas ---
# Esta línea crea las tablas definidas en models.py si no existen
models.Base.metadata.create_all(bind=engine)

# 1. Inicializar la aplicación FastAPI
app = FastAPI(title="API de Predicción de Fallas de Maquinaria v2.1 (con DB y CRUD)")

# --- Configuración de CORS ---
origins = [
    "http://localhost:5173",  # Frontend de Vite
    "http://localhost:3000",  # Frontend de Next.js/CRA
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Incluir el Router del CRUD ---
# Ahora tendrás endpoints como /api/machines/, /api/readings/{id}, etc.
app.include_router(crud_endpoints.router)


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

# 4. Endpoint de bienvenida
@app.get("/")
def bienvenida():
    return {"mensaje": "API del Doctor de Máquinas v2.1 está funcionando. Revisa /docs para la documentación."}

# 5. Endpoint de predicción (Actualizado para guardar en BD)
#    Usa los schemas importados para la entrada (DatosMaquinaPrediccion) y salida (PrediccionResponse)
@app.post("/predecir", response_model=schemas.PrediccionResponse)
def predecir_falla(datos: schemas.DatosMaquinaPrediccion, db: Session = Depends(get_db)):
    if not modelo_falla or not modelo_tipo_falla:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos no cargados. Revisa la consola del backend."
        )

    # Verifica que la máquina exista en la BD
    db_machine = db.query(models.Machine).filter(models.Machine.machine_id == datos.machine_id).first()
    if db_machine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Máquina con machine_id {datos.machine_id} no encontrada. Créala primero usando /api/machines/"
        )
    
    # Verifica que el 'Type' de los datos coincida con el de la máquina
    if db_machine.type != datos.Type:
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"El 'Type' {datos.Type} no coincide con el tipo '{db_machine.type}' de la máquina {datos.machine_id}."
        )

    # 6. Procesar los datos de entrada
    # Excluimos machine_id de los datos del modelo
    input_data = datos.model_dump(exclude={"machine_id"}) 
    input_df = pd.DataFrame([input_data])
    input_dummies = pd.get_dummies(input_df, columns=['Type'])
    
    input_final = pd.DataFrame(columns=columnas_modelo)
    # Concatena y rellena con 0 las columnas que no estaban en input_dummies
    input_final = pd.concat([input_final, input_dummies]).fillna(0)
    # Asegura el orden correcto de las columnas
    input_final = input_final[columnas_modelo] 

        # --- PASO B: PREDICCIÓN CON MODELO 1 (¿Hay Falla?) ---
        prediccion_falla = modelo_falla.predict(input_final)
        probabilidad_falla = modelo_falla.predict_proba(input_final)
        
        resultado_falla = int(prediccion_falla[0])
        confianza = float(probabilidad_falla[0][resultado_falla]) * 100

        hubo_falla_bool = (resultado_falla == 1)

        # --- Guardar la LECTURA en la base de datos ---
        # (Tu código ya hacía esto correctamente)
        db_reading = models.MachineReading(
            machine_id=datos.machine_id,
            air_temperature=datos.temp_aire,
            process_temperature=datos.temp_proceso,
            rotational_speed=datos.velocidad_rotacion,
            torque=datos.torque,
            tool_wear=datos.desgaste_herramienta,
            machine_failure=hubo_falla_bool
        )
        db.add(db_reading)
        db.commit()
        db.refresh(db_reading) # <-- ¡IMPORTANTE! Para obtener el reading_id generado


        # 7. Decidir la respuesta
        if not hubo_falla_bool:
            # --- CASO: OPERACIÓN NORMAL ---
            return {
                "prediccion": "OPERACION NORMAL",
                "confianza": f"{confianza:.2f}%",
                "tipo_falla_probable": "N/A",
                "recomendacion": "Continuar operación estándar.",
                "reading_saved_id": db_reading.reading_id
            }
        
        else:
            # --- CASO: FALLA PROBABLE ---
            prediccion_tipo = modelo_tipo_falla.predict(input_final)
            
            falla_str_lista = []
            rec_str_lista = []
            
            # Inicializa el dict de fallas en False
            detalles_falla_dict = {label.lower(): False for label in labels_tipo_falla} # {'twf': False, ...}
            detalles_falla_dict["rnf"] = False # Añadir random failure por si acaso

            for i, label in enumerate(labels_tipo_falla): # ['TWF', 'HDF', 'PWF', 'OSF']
                if prediccion_tipo[0][i] == 1:
                    recomendacion = RECOMENDACIONES.get(label, "Revisión requerida.")
                    falla_str_lista.append(recomendacion.split('.')[0])
                    rec_str_lista.append(recomendacion.split('.')[1].strip())
                    detalles_falla_dict[label.lower()] = True # Marca la falla como True

            if not falla_str_lista:
                tipo_falla_str = RECOMENDACIONES["OTRA"].split('.')[0]
                recomendacion_str = RECOMENDACIONES["OTRA"].split('.')[1].strip()
                detalles_falla_dict["rnf"] = True # Marcar como falla random
            else:
                tipo_falla_str = ", ".join(falla_str_lista)
                recomendacion_str = " ".join(rec_str_lista)

            # --- Guardar los DETALLES DE FALLA en la base de datos ---
            # (Tu código ya hacía esto correctamente)
            db_failure_details = models.FailureType(
                reading_id=db_reading.reading_id,
                **detalles_falla_dict # Desempaqueta el dict: twf=True, hdf=False, ...
            )
            db.add(db_failure_details)
            db.commit()

            return {
                "prediccion": "FALLA PROBABLE",
                "confianza": f"{confianza:.2f}%",
                "tipo_falla_probable": tipo_falla_str, # Corregido el typo
                "recomendacion": recomendacion_str,
                "reading_saved_id": db_reading.reading_id
            }

    # Capturar cualquier error inesperado durante la predicción
    except Exception as e:
        db.rollback() # Deshacer cualquier cambio en la BD si la predicción falla
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la predicción: {str(e)}"
        )

