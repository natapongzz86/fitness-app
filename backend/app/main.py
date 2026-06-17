"""
จุดเริ่มต้นของ FastAPI Backend
รัน: uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base
from app.routers import auth, workout

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Fitness Trainer API",
    description="ระบบช่วยออกกำลังกายด้วย Image Processing (MediaPipe Pose + Custom CNN)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ใน production ควรระบุ origin ของ Streamlit ให้ชัดเจน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(workout.router)


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Fitness Trainer API กำลังทำงาน"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
