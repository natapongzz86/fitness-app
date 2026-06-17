from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.db_models import User, WorkoutSession
from app.models.schemas import WorkoutSessionCreate, WorkoutSessionOut
from app.utils.security import get_current_user
from app.utils import pose_processing

router = APIRouter(prefix="/workout", tags=["workout"])


@router.post("/analyze-frame")
async def analyze_frame(
    file: UploadFile = File(...),
    exercise_type: str = Form("squat"),
    session_id: str = Form(...),
    current_user: User = Depends(get_current_user),
):
    image_bytes = await file.read()
    result = pose_processing.analyze_frame(image_bytes, exercise_type, session_id)
    return result


@router.post("/reset-session")
def reset_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    pose_processing.reset_session(session_id)
    return {"status": "reset"}


@router.post("/sessions", response_model=WorkoutSessionOut)
def save_session(
    payload: WorkoutSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = WorkoutSession(
        user_id=current_user.id,
        exercise_type=payload.exercise_type,
        rep_count=payload.rep_count,
        accuracy_score=payload.accuracy_score,
        duration_seconds=payload.duration_seconds,
        calories_estimate=payload.calories_estimate,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=List[WorkoutSessionOut])
def get_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(WorkoutSession)
        .filter(WorkoutSession.user_id == current_user.id)
        .order_by(WorkoutSession.created_at.desc())
        .all()
    )