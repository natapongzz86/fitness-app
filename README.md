# AI Fitness Trainer — ระบบช่วยออกกำลังกายด้วย Image Processing

ระบบประกอบด้วย 3 ส่วน:

| ส่วน | เทคโนโลยี | หน้าที่ |
|---|---|---|
| **Backend** | Python, FastAPI, MediaPipe, OpenCV, TensorFlow | รับภาพจากกล้อง วิเคราะห์ pose, นับ rep, ตรวจฟอร์ม, ระบบล็อกอิน (JWT), เก็บประวัติ |
| **Frontend** | Streamlit, streamlit-webrtc | หน้าล็อกอิน/สมัครสมาชิก + แดชบอร์ดออกกำลังกายแบบ real-time ผ่านเว็บแคม |
| **Colab** | Google Colab, TensorFlow, MediaPipe | เทรนโมเดลจำแนกฟอร์มท่าออกกำลังกายจากชุดข้อมูลภาพ 10,000+ รูป |

## โครงสร้างโปรเจกต์

```
fitness-app/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint
│   │   ├── database.py          # ตั้งค่า SQLite + SQLAlchemy
│   │   ├── models/
│   │   │   ├── db_models.py     # ORM: User, WorkoutSession
│   │   │   └── schemas.py       # Pydantic schemas
│   │   ├── routers/
│   │   │   ├── auth.py          # สมัครสมาชิก/ล็อกอิน (JWT)
│   │   │   └── workout.py       # วิเคราะห์ภาพ + บันทึกประวัติ
│   │   ├── utils/
│   │   │   ├── security.py      # hash password, JWT
│   │   │   └── pose_processing.py  # MediaPipe pose detection + นับ rep
│   │   └── ml_models/           # วางไฟล์ .h5 ที่เทรนจาก Colab ไว้ที่นี่
│   └── requirements.txt
├── frontend/
│   ├── app.py                   # Streamlit: หน้าล็อกอิน + แดชบอร์ด
│   └── requirements.txt
└── colab/
    └── train_pose_classifier.ipynb   # เทรนโมเดลจาก dataset 10,000+ รูป
```

## วิธีรัน

### 1. ติดตั้งและรัน Backend (FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
ทดสอบ API ได้ที่ `http://localhost:8000/docs` (Swagger UI อัตโนมัติ)

### 2. ติดตั้งและรัน Frontend (Streamlit)
เปิด terminal ใหม่ (เก็บ backend ให้รันต่อไป):
```bash
cd frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
เปิดเบราว์เซอร์ที่ `http://localhost:8501`

### 3. เทรนโมเดลฟอร์มออกกำลังกายบน Google Colab
1. อัปโหลด `colab/train_pose_classifier.ipynb` เข้า Google Colab (หรือเปิดผ่าน Google Drive)
2. เตรียม dataset (10,000+ รูป) ใน Google Drive ตามโครงสร้างที่ระบุในเซลล์แรกของ notebook
3. เลือก Runtime → Change runtime type → GPU
4. รันทุกเซลล์ตามลำดับ ระบบจะ:
   - แตกจุด landmark ร่างกายจากทุกภาพด้วย MediaPipe Pose
   - เทรน Neural Network จำแนกฟอร์มถูก/ผิด
   - บันทึกโมเดลเป็น `.h5` และ `.tflite` ลง Google Drive
5. ดาวน์โหลดไฟล์ `*_form_classifier.h5` แล้ววางไว้ที่ `backend/app/ml_models/exercise_form_classifier.h5`
6. รีสตาร์ท backend — ระบบจะโหลดโมเดลที่เทรนเองมาใช้แทนการประเมินแบบ rule-based โดยอัตโนมัติ

## ฟีเจอร์หลัก

- **ระบบสมาชิก**: สมัครสมาชิก / เข้าสู่ระบบ พร้อม JWT authentication และ bcrypt password hashing
- **วิเคราะห์ท่าทางแบบ real-time**: ใช้ MediaPipe Pose (33 landmark points) ตรวจจับร่างกายจากเว็บแคมผ่าน streamlit-webrtc
- **นับจำนวนครั้งอัตโนมัติ**: คำนวณมุมข้อต่อ (hip-knee-ankle สำหรับ squat, shoulder-elbow-wrist สำหรับ push-up) เพื่อนับ rep และตรวจ stage (up/down)
- **ตรวจฟอร์มและให้คำแนะนำ**: ทั้งแบบ rule-based (มุมข้อต่อ) และแบบโมเดล AI ที่เทรนเอง (รองรับการสวอปเข้าทันทีที่มีไฟล์ .h5)
- **บันทึกและดูประวัติ**: เก็บผลแต่ละครั้งลง SQLite พร้อมกราฟสรุปและตารางประวัติใน Streamlit

## หมายเหตุด้านความปลอดภัย

ก่อนนำไป production ควร: เปลี่ยน `JWT_SECRET_KEY` ใน `backend/app/utils/security.py` เป็นค่าสุ่มที่เก็บใน environment variable, จำกัด CORS origins ให้เฉพาะ domain ของ Streamlit จริง, และเปลี่ยนจาก SQLite เป็น PostgreSQL หากต้องรองรับผู้ใช้จำนวนมาก
