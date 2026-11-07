"""
MaintiDoc - Database Connection and Health Module
Manages PostgreSQL connections, session scopes, and schema initialization.
"""

from typing import Generator, Dict, Any, Tuple
from contextlib import contextmanager
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from config import settings, setup_logger

logger = setup_logger("database.connection")

# Create SQLAlchemy engine with connection pool
engine = create_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional database session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session rollback due to error: {e}", exc_info=True)
        raise
    finally:
        session.close()


def check_db_health() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Checks if PostgreSQL is reachable and pgvector extension is available.
    Returns: (is_healthy, status_message, details_dict)
    """
    details: Dict[str, Any] = {
        "host": settings.POSTGRES_HOST,
        "port": settings.POSTGRES_PORT,
        "database": settings.POSTGRES_DB,
        "pg_version": None,
        "pgvector_installed": False,
        "tables": [],
    }

    try:
        with engine.connect() as conn:
            # 1. Check PostgreSQL version
            version_result = conn.execute(text("SELECT version();")).scalar()
            details["pg_version"] = version_result

            # 2. Check if vector extension is installed
            ext_result = conn.execute(
                text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
            ).fetchone()

            if ext_result:
                details["pgvector_installed"] = True
                details["pgvector_version"] = ext_result[1]
            else:
                details["pgvector_installed"] = False

            # 3. Check existing tables
            tables_result = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' ORDER BY table_name;"
                )
            ).fetchall()
            details["tables"] = [row[0] for row in tables_result]

            return True, "PostgreSQL & PGVector connection successful", details

    except SQLAlchemyError as sqle:
        logger.warning(f"Database health check failed (SQLAlchemyError): {sqle}")
        return False, f"PostgreSQL connection error: {str(sqle)}", details
    except Exception as exc:
        logger.warning(f"Database health check failed: {exc}")
        return False, f"Connection failed: {str(exc)}", details


def init_database() -> bool:
    """
    Executes database/schema.sql to create extensions, tables, and indices.
    Returns True if successful, False otherwise.
    """
    schema_path = Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        logger.error(f"Schema file not found at {schema_path}")
        return False

    try:
        sql_content = schema_path.read_text(encoding="utf-8")
        with engine.connect() as conn:
            # Split and execute individual statements to handle transactions properly
            # Or execute as raw text block with autocommit
            with conn.begin():
                conn.execute(text(sql_content))
        logger.info("Database schema initialized successfully.")
        return True
    except Exception as exc:
        logger.error(f"Failed to initialize database schema: {exc}", exc_info=True)
        return False
