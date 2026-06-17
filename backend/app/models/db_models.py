"""
โมเดลฐานข้อมูล: ผู้ใช้งาน และ ประวัติการออกกำลังกาย
"""
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("WorkoutSession", back_populates="owner")


class WorkoutSession(Base):
    __tablename__ = "workout_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exercise_type = Column(String, nullable=False)   # เช่น squat, pushup, plank
    rep_count = Column(Integer, default=0)
    accuracy_score = Column(Float, default=0.0)       # ความแม่นยำของฟอร์ม (0-100)
    duration_seconds = Column(Float, default=0.0)
    calories_estimate = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="sessions")
