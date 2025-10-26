from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# --- Schemas para FailureType ---
class FailureTypeBase(BaseModel):
    twf: bool = False
    hdf: bool = False
    pwf: bool = False
    osf: bool = False
    rnf: bool = False

class FailureTypeCreate(FailureTypeBase):
    pass

class FailureTypeResponse(FailureTypeBase):
    failure_id: int
    reading_id: int

    class Config:
        from_attributes = True # Nuevo Pydantic v2 (reemplaza orm_mode)

# --- Schemas para MachineReading ---
class MachineReadingBase(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int
    machine_failure: bool = False

class MachineReadingCreate(MachineReadingBase):
    pass

class MachineReadingResponse(MachineReadingBase):
    reading_id: int
    machine_id: int
    timestamp: datetime
    # Opcionalmente, incluir los detalles de la falla si existen
    failure_details: Optional[FailureTypeResponse] = None

    class Config:
        from_attributes = True

# --- Schemas para Machine ---
class MachineBase(BaseModel):
    type: str
    location: Optional[str] = None
    description: Optional[str] = None

class MachineCreate(MachineBase):
    pass

class MachineResponse(MachineBase):
    machine_id: int
    # Incluir una lista de las lecturas asociadas
    readings: List[MachineReadingResponse] = []

    class Config:
        from_attributes = True

# --- Schema para la ENTRADA de predicción ---
# (Requerido por main.py)
class DatosMaquinaPrediccion(BaseModel):
    machine_id: int # ID de la máquina a la que pertenece esta lectura
    temp_aire: float
    temp_proceso: float
    velocidad_rotacion: int
    torque: float
    desgaste_herramienta: int
    Type: str  # "L", "M", o "H"

# --- Schema para la RESPUESTA de predicción ---
# (Requerido por main.py)
class PrediccionResponse(BaseModel):
    prediccion: str
    confianza: str
    tipo_falla_probable: str
    recomendacion: str
    reading_saved_id: int

