from sqlalchemy import text
from app.core.logging import logger

def run_db_migrations(engine):
    """
    Runs schema checks and migrations on startup.
    Ensures that the 'processing_time' column exists in the analyses table.
    """
    try:
        with engine.begin() as conn:
            # Check table columns using SQLite PRAGMA
            columns_query = conn.execute(text("PRAGMA table_info(analyses)"))
            columns = [row[1] for row in columns_query.fetchall()]
            
            # If the table exists but the column is missing, run migration
            if columns and "processing_time" not in columns:
                logger.info("Migrating database: Adding 'processing_time' column to analyses table...")
                conn.execute(text("ALTER TABLE analyses ADD COLUMN processing_time FLOAT DEFAULT 0.0"))
                logger.info("Database migration complete.")
    except Exception as e:
        logger.error(f"Failed to run database migrations: {e}")
