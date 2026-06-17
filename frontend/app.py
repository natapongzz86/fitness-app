"""
Streamlit Frontend: หน้าล็อกอิน/สมัครสมาชิก + แดชบอร์ดออกกำลังกายแบบ real-time
รัน: streamlit run app.py
"""
import streamlit as st
import requests
import cv2
import numpy as np
import av
import pandas as pd
import uuid
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Fitness Trainer", page_icon="💪", layout="wide")

# ---------------- Session State ----------------
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "workout_session_id" not in st.session_state:
    st.session_state.workout_session_id = str(uuid.uuid4())


def api_post(path, json=None, data=None, files=None, auth=True):
    headers = {}
    if auth and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    return requests.post(f"{API_BASE_URL}{path}", json=json, data=data, files=files, headers=headers)


def api_get(path, auth=True):
    headers = {}
    if auth and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    return requests.get(f"{API_BASE_URL}{path}", headers=headers)


def safe_get_detail(resp, default="เกิดข้อผิดพลาด"):
    """แปลง response เป็น JSON อย่างปลอดภัย ป้องกัน JSONDecodeError"""
    try:
        return resp.json().get("detail", default)
    except Exception:
        return f"{default} (server responded with status {resp.status_code})"


# ===================================================================
# หน้าล็อกอิน / สมัครสมาชิก
# ===================================================================
def render_login_page():
    st.title("💪 AI Fitness Trainer")
    st.caption("ระบบช่วยออกกำลังกายด้วย Image Processing")

    tab_login, tab_register = st.tabs(["เข้าสู่ระบบ", "สมัครสมาชิก"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("ชื่อผู้ใช้", key="login_username")
            password = st.text_input("รหัสผ่าน", type="password", key="login_password")
            submitted = st.form_submit_button("เข้าสู่ระบบ", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.warning("กรุณากรอกชื่อผู้ใช้และรหัสผ่าน")
                else:
                    try:
                        resp = api_post(
                            "/auth/login",
                            json={"username": username, "password": password},
                            auth=False,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.token = data["access_token"]
                            st.session_state.username = username
                            st.rerun()
                        else:
                            st.error(safe_get_detail(resp, "เข้าสู่ระบบไม่สำเร็จ"))
                    except requests.exceptions.ConnectionError:
                        st.error("ไม่สามารถเชื่อมต่อ Backend ได้ กรุณาตรวจสอบว่า FastAPI server กำลังรันอยู่ที่ port 8000")
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("ชื่อผู้ใช้")
            new_email = st.text_input("อีเมล")
            new_full_name = st.text_input("ชื่อ-นามสกุล (ไม่บังคับ)")
            new_password = st.text_input("รหัสผ่าน", type="password")
            confirm_password = st.text_input("ยืนยันรหัสผ่าน", type="password")
            submitted = st.form_submit_button("สมัครสมาชิก", use_container_width=True)

            if submitted:
                if new_password != confirm_password:
                    st.error("รหัสผ่านไม่ตรงกัน")
                elif not all([new_username, new_email, new_password]):
                    st.warning("กรุณากรอกข้อมูลให้ครบ")
                else:
                    try:
                        resp = api_post(
                            "/auth/register",
                            json={
                                "username": new_username,
                                "email": new_email,
                                "password": new_password,
                                "full_name": new_full_name or None,
                            },
                            auth=False,
                        )
                        if resp.status_code == 201:
                            st.success("สมัครสมาชิกสำเร็จ! กรุณาเข้าสู่ระบบที่แท็บ 'เข้าสู่ระบบ'")
                        else:
                            st.error(safe_get_detail(resp, "สมัครสมาชิกไม่สำเร็จ"))
                    except requests.exceptions.ConnectionError:
                        st.error("ไม่สามารถเชื่อมต่อ Backend ได้")
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")


# ===================================================================
# ตัวประมวลผลวิดีโอแบบ real-time (ส่งแต่ละเฟรมไปวิเคราะห์ที่ FastAPI)
# ===================================================================
class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.exercise_type = "squat"
        self.session_id = st.session_state.workout_session_id
        self.token = st.session_state.token
        self.last_result = {"rep_count": 0, "form_feedback": "", "accuracy_score": 0}
        self.last_call_time = 0
        self.call_interval = 0.3  # วิเคราะห์ทุก 0.3 วินาที เพื่อลดโหลด backend

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if now - self.last_call_time >= self.call_interval:
            self.last_call_time = now
            try:
                _, encoded = cv2.imencode(".jpg", img)
                resp = requests.post(
                    f"{API_BASE_URL}/workout/analyze-frame",
                    files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
                    data={"exercise_type": self.exercise_type, "session_id": self.session_id},
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=2,
                )
                if resp.status_code == 200:
                    self.last_result = resp.json()
            except Exception:
                pass

        # วาดข้อมูลซ้อนบนภาพ
        cv2.putText(img, f"Reps: {self.last_result.get('rep_count', 0)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(img, self.last_result.get("form_feedback", "")[:40], (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ===================================================================
# หน้าแดชบอร์ดหลัก (หลังล็อกอิน)
# ===================================================================
def render_dashboard():
    with st.sidebar:
        st.success(f"สวัสดี, {st.session_state.username} 👋")
        if st.button("ออกจากระบบ", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()
        st.divider()
        page = st.radio("เมนู", ["ออกกำลังกาย (Real-time)", "ประวัติการออกกำลังกาย"])

    if page == "ออกกำลังกาย (Real-time)":
        render_workout_page()
    else:
        render_history_page()


def render_workout_page():
    st.title("🏋️ เริ่มออกกำลังกาย")

    col1, col2 = st.columns([1, 1])
    with col1:
        exercise_type = st.selectbox(
            "เลือกท่าออกกำลังกาย",
            options=["squat", "pushup", "plank"],
            format_func=lambda x: {"squat": "สควอท (Squat)", "pushup": "วิดพื้น (Push-up)", "plank": "แพลงก์ (Plank)"}[x],
        )
    with col2:
        if st.button("🔄 รีเซ็ตตัวนับ"):
            try:
                requests.post(
                    f"{API_BASE_URL}/workout/reset-session",
                    params={"session_id": st.session_state.workout_session_id},
                    headers={"Authorization": f"Bearer {st.session_state.token}"},
                )
            except Exception:
                pass
            st.session_state.workout_session_id = str(uuid.uuid4())
            st.success("รีเซ็ตเรียบร้อย")

    st.info("กดอนุญาตให้เว็บเบราว์เซอร์ใช้กล้อง แล้วทำท่าออกกำลังกายตรงหน้ากล้องให้เห็นเต็มตัว")

    ctx = webrtc_streamer(
        key="pose-workout",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PoseVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.exercise_type = exercise_type

    st.divider()
    st.subheader("💾 บันทึกผลการออกกำลังกายครั้งนี้")
    with st.form("save_session_form"):
        c1, c2, c3 = st.columns(3)
        rep_count = c1.number_input("จำนวนครั้ง (reps)", min_value=0, value=0)
        duration = c2.number_input("ระยะเวลา (วินาที)", min_value=0.0, value=0.0)
        calories = c3.number_input("แคลอรี่ที่เผาผลาญ (ประมาณ)", min_value=0.0, value=0.0)
        accuracy = st.slider("ความแม่นยำของฟอร์มโดยรวม (%)", 0, 100, 80)

        if st.form_submit_button("บันทึกผล", use_container_width=True):
            resp = api_post(
                "/workout/sessions",
                json={
                    "exercise_type": exercise_type,
                    "rep_count": int(rep_count),
                    "accuracy_score": float(accuracy),
                    "duration_seconds": float(duration),
                    "calories_estimate": float(calories),
                },
            )
            if resp.status_code == 200:
                st.success("บันทึกผลสำเร็จ!")
            else:
                st.error("บันทึกผลไม่สำเร็จ")


def render_history_page():
    st.title("📊 ประวัติการออกกำลังกาย")
    resp = api_get("/workout/sessions")
    if resp.status_code != 200:
        st.error("ไม่สามารถดึงประวัติได้")
        return

    sessions = resp.json()
    if not sessions:
        st.info("ยังไม่มีประวัติการออกกำลังกาย")
        return

    df = pd.DataFrame(sessions)
    df["created_at"] = pd.to_datetime(df["created_at"])

    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนครั้งทั้งหมด", int(df["rep_count"].sum()))
    c2.metric("แคลอรี่สะสม", f"{df['calories_estimate'].sum():.0f} kcal")
    c3.metric("ความแม่นยำเฉลี่ย", f"{df['accuracy_score'].mean():.1f}%")

    st.line_chart(df.set_index("created_at")[["rep_count", "accuracy_score"]])
    st.dataframe(
        df[["created_at", "exercise_type", "rep_count", "accuracy_score", "duration_seconds", "calories_estimate"]],
        use_container_width=True,
    )


# ===================================================================
# Main
# ===================================================================
if st.session_state.token is None:
    render_login_page()
else:
    render_dashboard()
