from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ---------- Auth ----------
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ---------- Workout ----------
class WorkoutSessionCreate(BaseModel):
    exercise_type: str
    rep_count: int = 0
    accuracy_score: float = 0.0
    duration_seconds: float = 0.0
    calories_estimate: float = 0.0


class WorkoutSessionOut(BaseModel):
    id: int
    exercise_type: str
    rep_count: int
    accuracy_score: float
    duration_seconds: float
    calories_estimate: float
    created_at: datetime

    class Config:
        from_attributes = True