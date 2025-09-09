from sqlalchemy import Column, Integer, String, Float, DateTime,create_engine, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
# already
# db_models.py
# endalready
import os
 
#  alreadycommentes
# from datetime import datetime
# from .database import Base 
# class Detection(Base):
#  tablename = "detections"
#  id = Column(Integer, primary_key=True, index=True)
# class_name = Column(String, nullable=False)
# confidence = Column(Float, nullable=False)
# bbox = Column(Text, nullable=False)
# filename = Column(String, nullable=False)
# username = Column(String, nullable=False)
# timestamp = Column(DateTime, default=datetime.utcnow)

# Base and Engine setup
# DATABASE_URL = "sqlite:///./users.db"  # single DB
# endalready
DB_FILENAME = os.getenv("SQLITE_DB", "users_new.db")  # change default name if you like
DATABASE_URL = "sqlite:///./new_database.db"  # ← just change the file name


# againalreadycommentes
# Base = declarative_base()
# engine = create_engine("sqlite:///inventory.db", connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(bind=engine)

# engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False, "timeout": 30})
# endalready
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False,bind=engine)
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
# already
    # detections = relationship("Detection", back_populates="user")
# endalready

# Detection model
class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    username = Column(String)  # ✅ NEW: store username directly
    class_name = Column(String)
    confidence = Column(Float)
    bbox = Column(String)
    filename = Column(String)
    timestamp = Column(DateTime, default=datetime.now)
    # already
    # ✅ Only keep this:
    # user_id = Column(Integer, ForeignKey('users.id'))
    # username = Column(String)  # ✅ NEW: store username directly

    # user = relationship("User", back_populates="detections")
    # endalready

class Upload(Base):
    __tablename__ = "uploads"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    uploaded_by = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

# Create tables
Base.metadata.create_all(bind=engine)
