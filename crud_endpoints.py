from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

# Importa los modelos, schemas y el get_db
import models
import schemas
from database import get_db

# Crea un "mini-FastAPI" para agrupar estos endpoints
router = APIRouter(
    prefix="/api",  # Todos los endpoints aquí empezarán con /api
    tags=["CRUD Management"] # Se agruparán en Swagger bajo "CRUD Management"
)

# ================================
# CRUD para Machines (Máquinas)
# ================================

@router.post("/machines/", response_model=schemas.MachineResponse, status_code=status.HTTP_201_CREATED)
def create_machine(machine: schemas.MachineCreate, db: Session = Depends(get_db)):
    """
    CREATE: Crea una nueva máquina.
    """
    db_machine = models.Machine(**machine.model_dump())
    db.add(db_machine)
    db.commit()
    db.refresh(db_machine)
    return db_machine

@router.get("/machines/{machine_id}", response_model=schemas.MachineResponse)
def read_machine(machine_id: int, db: Session = Depends(get_db)):
    """
    READ (One): Obtiene una máquina específica por su ID.
    """
    db_machine = db.query(models.Machine).filter(models.Machine.machine_id == machine_id).first()
    if db_machine is None:
        raise HTTPException(status_code=404, detail="Machine not found")
    return db_machine

@router.get("/machines/", response_model=List[schemas.MachineResponse])
def read_machines(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    READ (All): Obtiene una lista de todas las máquinas.
    """
    machines = db.query(models.Machine).offset(skip).limit(limit).all()
    return machines

@router.put("/machines/{machine_id}", response_model=schemas.MachineResponse)
def update_machine(machine_id: int, machine: schemas.MachineCreate, db: Session = Depends(get_db)):
    """
    UPDATE: Actualiza la información de una máquina existente.
    """
    db_machine = read_machine(machine_id, db) # Reutiliza la función read_machine para obtener y chequear 404
    
    # Actualiza los campos
    db_machine.type = machine.type
    db_machine.location = machine.location
    db_machine.description = machine.description
    
    db.commit()
    db.refresh(db_machine)
    return db_machine

@router.delete("/machines/{machine_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_machine(machine_id: int, db: Session = Depends(get_db)):
    """
    DELETE: Elimina una máquina. (Las lecturas se borrarán en cascada por el ON DELETE CASCADE)
    """
    db_machine = read_machine(machine_id, db) # Obtener y chequear 404
    db.delete(db_machine)
    db.commit()
    return

# ================================
# CRUD para Machine Readings (Lecturas)
# ================================
# Nota: El endpoint /predecir ya funciona como un "CREATE" de lecturas.
# Estos son endpoints adicionales para gestión manual.

@router.get("/readings/{reading_id}", response_model=schemas.MachineReadingResponse)
def read_reading(reading_id: int, db: Session = Depends(get_db)):
    """
    READ (One): Obtiene una lectura específica por su ID.
    """
    db_reading = db.query(models.MachineReading).filter(models.MachineReading.reading_id == reading_id).first()
    if db_reading is None:
        raise HTTPException(status_code=404, detail="Reading not found")
    return db_reading

@router.get("/machines/{machine_id}/readings/", response_model=List[schemas.MachineReadingResponse])
def read_readings_for_machine(machine_id: int, skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """
    READ (All): Obtiene todas las lecturas de una máquina específica.
    """
    # Primero, verifica que la máquina exista
    read_machine(machine_id, db)
    
    readings = db.query(models.MachineReading)\
                 .filter(models.MachineReading.machine_id == machine_id)\
                 .offset(skip).limit(limit).all()
    return readings

@router.delete("/readings/{reading_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_reading(reading_id: int, db: Session = Depends(get_db)):
    """
    DELETE: Elimina una lectura. (El detalle de falla se borrará en cascada)
    """
    db_reading = read_reading(reading_id, db) # Obtener y chequear 404
    db.delete(db_reading)
    db.commit()
    return

# ================================
# CRUD para Failure Types (Tipos de Falla)
# ================================
# Nota: El endpoint /predecir ya funciona como un "CREATE" de fallas.
# Estos endpoints son para LEER o CORREGIR (Update/Delete) una falla.

@router.get("/readings/{reading_id}/failure_details/", response_model=schemas.FailureTypeResponse)
def read_failure_for_reading(reading_id: int, db: Session = Depends(get_db)):
    """
    READ (One): Obtiene el detalle de falla asociado a una lectura específica.
    """
    # Primero, verifica que la lectura exista
    read_reading(reading_id, db)
    
    db_failure = db.query(models.FailureType).filter(models.FailureType.reading_id == reading_id).first()
    
    if db_failure is None:
        raise HTTPException(status_code=404, detail="No failure details found for this reading")
    return db_failure

@router.put("/failure_types/{failure_id}", response_model=schemas.FailureTypeResponse)
def update_failure_type(failure_id: int, failure: schemas.FailureTypeCreate, db: Session = Depends(get_db)):
    """
    UPDATE: Actualiza/corrige un detalle de falla (ej. si el ML se equivocó).
    """
    db_failure = db.query(models.FailureType).filter(models.FailureType.failure_id == failure_id).first()
    
    if db_failure is None:
        raise HTTPException(status_code=404, detail="FailureType record not found")

    # Actualiza todos los campos
    db_failure.twf = failure.twf
    db_failure.hdf = failure.hdf
    db_failure.pwf = failure.pwf
    db_failure.osf = failure.osf
    db_failure.rnf = failure.rnf
    
    db.commit()
    db.refresh(db_failure)
    return db_failure

@router.delete("/failure_types/{failure_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_failure_type(failure_id: int, db: Session = Depends(get_db)):
    """
    DELETE: Elimina un registro de detalle de falla.
    """
    db_failure = db.query(models.FailureType).filter(models.FailureType.failure_id == failure_id).first()
    
    if db_failure is None:
        raise HTTPException(status_code=404, detail="FailureType record not found")

    db.delete(db_failure)
    db.commit()
    return
