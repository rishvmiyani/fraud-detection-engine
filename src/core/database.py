"""
Database Connection and Session Management
Handles PostgreSQL connection, session management, and database operations
"""

import databases
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager, contextmanager
import logging
from typing import AsyncGenerator, Generator

from .config import get_settings, get_database_settings

logger = logging.getLogger(__name__)
settings = get_settings()
db_settings = get_database_settings()

# SQLAlchemy setup
engine = sqlalchemy.create_engine(
    settings.DATABASE_URL,
    **db_settings.connection_args
)

# Async database interface
database = databases.Database(
    settings.DATABASE_URL,
    **db_settings.async_connection_args
)

# Session maker for synchronous operations
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()


# Dependency for getting async database connection
async def get_database() -> databases.Database:
    """Get async database connection"""
    return database


# Dependency for getting sync database session
def get_db_session():
    """Get synchronous database session"""
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()


# Context manager for async database operations
@asynccontextmanager
async def get_async_session() -> AsyncGenerator[databases.Database, None]:
    """Async context manager for database operations"""
    try:
        yield database
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise


# Context manager for sync database operations
@contextmanager
def get_sync_session() -> Generator[sqlalchemy.orm.Session, None, None]:
    """Sync context manager for database operations"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database operation failed: {e}")
        raise
    finally:
        session.close()


# Database initialization
async def init_database():
    """Initialize database tables"""
    try:
        # Import all models to register them
        from models import user, transaction, alert, model_metadata
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


# Database connection management
async def connect_database():
    """Connect to database"""
    try:
        await database.connect()
        logger.info("Connected to database successfully")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


async def disconnect_database():
    """Disconnect from database"""
    try:
        await database.disconnect()
        logger.info("Disconnected from database")
    except Exception as e:
        logger.error(f"Database disconnection failed: {e}")


# Health check function
async def check_database_health() -> bool:
    """Check if database is healthy"""
    try:
        # Simple query to test connection
        query = "SELECT 1"
        result = await database.fetch_one(query)
        return result is not None
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# Database utility functions
class DatabaseManager:
    """Database management utility class"""
    
    @staticmethod
    async def execute_query(query: str, values: dict = None):
        """Execute a query with optional parameters"""
        try:
            if values:
                result = await database.execute(query, values)
            else:
                result = await database.execute(query)
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    @staticmethod
    async def fetch_one(query: str, values: dict = None):
        """Fetch single row"""
        try:
            if values:
                result = await database.fetch_one(query, values)
            else:
                result = await database.fetch_one(query)
            return result
        except Exception as e:
            logger.error(f"Fetch one failed: {e}")
            raise
    
    @staticmethod
    async def fetch_all(query: str, values: dict = None):
        """Fetch multiple rows"""
        try:
            if values:
                result = await database.fetch_all(query, values)
            else:
                result = await database.fetch_all(query)
            return result
        except Exception as e:
            logger.error(f"Fetch all failed: {e}")
            raise
    
    @staticmethod
    async def execute_transaction(queries_and_values: list):
        """Execute multiple queries in a transaction"""
        transaction = await database.transaction()
        try:
            for query, values in queries_and_values:
                if values:
                    await database.execute(query, values)
                else:
                    await database.execute(query)
            
            await transaction.commit()
            logger.info("Transaction completed successfully")
            
        except Exception as e:
            await transaction.rollback()
            logger.error(f"Transaction failed and rolled back: {e}")
            raise
    
    @staticmethod
    def create_sync_session():
        """Create synchronous session for background tasks"""
        return SessionLocal()


# Database connection instance
db_manager = DatabaseManager()
