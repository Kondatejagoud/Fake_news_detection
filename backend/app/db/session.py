from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings
from app.core.logging import logger

# Replace postgres:// with postgresql:// if needed (Railway compatibility)
db_url = settings.DATABASE_URL
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

logger.info(f"Connecting to database (sanitized schema: {db_url.split('://')[0]}://...)")

# Configure connection pooling and timeout settings for stability
connect_args = {}
if db_url.startswith("sqlite"):
    # SQLite requires check_same_thread=False for multi-threaded FastAPI access
    connect_args = {"check_same_thread": False}
else:
    # PostgreSQL pooling settings
    connect_args = {
        "sslmode": "prefer"
    }

try:
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args=connect_args
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.error(f"Failed to initialize database engine for url {db_url.split('://')[0]}...: {e}")
    # Force fallback to SQLite local database if initialization fails
    logger.info("Falling back to local SQLite database: sqlite:///./fake_detection.db")
    engine = create_engine(
        "sqlite:///./fake_detection.db",
        pool_pre_ping=True,
        connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# FastAPI db session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
