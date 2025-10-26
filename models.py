from sqlalchemy import create_engine, Column, Integer, String, Text, Numeric, Boolean, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from database import Base  # Importamos la Base de database.py

# ================================
#  TABLA: machines (máquinas)
# ================================
class Machine(Base):
    __tablename__ = "machines"

    machine_id = Column(Integer, primary_key=True, index=True) # SERIAL se maneja como Integer
    type = Column(String(50), nullable=False)
    location = Column(String(100))
    description = Column(Text, nullable=True)

    # RELACIÓN: Una máquina tiene muchas lecturas
    readings = relationship("MachineReading", back_populates="machine")

# =============================================
#  TABLA: machine_readings (lecturas de sensores)
# =============================================
class MachineReading(Base):
    __tablename__ = "machine_readings"

    reading_id = Column(Integer, primary_key=True, index=True)
    
    # Llave foránea que apunta a la tabla 'machines'
    machine_id = Column(Integer, ForeignKey("machines.machine_id", ondelete="CASCADE"), nullable=False)
    
    air_temperature = Column(Numeric(6, 2))
    process_temperature = Column(Numeric(6, 2))
    rotational_speed = Column(Integer)
    torque = Column(Numeric(6, 2))
    tool_wear = Column(Integer)
    machine_failure = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=func.now()) # func.now() es el CURRENT_TIMESTAMP

    # RELACIÓN: Esta lectura pertenece a una máquina
    machine = relationship("Machine", back_populates="readings")
    
    # RELACIÓN: Esta lectura puede tener UN detalle de falla
    failure_details = relationship("FailureType", back_populates="reading", uselist=False) # uselist=False = Uno-a-Uno

# ====================================
#  TABLA: failure_types (tipos de fallas)
# ====================================
class FailureType(Base):
    __tablename__ = "failure_types"

    failure_id = Column(Integer, primary_key=True, index=True)
    
    # Llave foránea que apunta a la tabla 'machine_readings'
    reading_id = Column(Integer, ForeignKey("machine_readings.reading_id", ondelete="CASCADE"), nullable=False, unique=True) # unique=True para relación 1-a-1
    
    twf = Column(Boolean, default=False) # Tool Wear Failure
    hdf = Column(Boolean, default=False) # Hydraulic Failure
    pwf = Column(Boolean, default=False) # Power Failure
    osf = Column(Boolean, default=False) # Overstrain Failure
    rnf = Column(Boolean, default=False) # Random Failure

    # RELACIÓN: Este detalle de falla pertenece a UNA lectura
    reading = relationship("MachineReading", back_populates="failure_details")
