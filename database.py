from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# --- Credenciales de la Base de Datos (Hardcoded) ---
# ¡IMPORTANTE! Reemplaza estos valores con tus credenciales reales si son diferentes.
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "InnovaIA"
DB_USER = "postgres"
DB_PASSWORD = "123"
# ======================================================================

print(f"Conectando a: Usuario='{DB_USER}', Host='{DB_HOST}', DB='{DB_NAME}'")

# --- Configuración de la Conexión a la Base de Datos ---
# Formato de la cadena de conexión para PostgreSQL con psycopg2
# postgresql+psycopg2://user:password@host:port/dbname
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Crear el motor de SQLAlchemy
# `pool_pre_ping=True` ayuda a manejar conexiones que se cierran por el servidor de DB
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Configurar SessionLocal para crear sesiones de base de datos
# `autocommit=False` y `autoflush=False` son configuraciones estándar para sesiones ORM
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Crear una clase base para los modelos declarativos de SQLAlchemy
Base = declarative_base()

# --- Dependencia para FastAPI ---
# Esta función generará una sesión de base de datos por cada solicitud HTTP
# y se asegurará de cerrarla correctamente.
def get_db():
    db = SessionLocal()
    try:
        yield db # Retorna la sesión al endpoint que la solicitó
    finally:
        db.close() # Asegura que la sesión se cierre después de la solicitud