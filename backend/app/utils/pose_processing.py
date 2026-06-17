"""
โมดูลประมวลผลภาพ (Image Processing) สำหรับวิเคราะห์ท่าออกกำลังกาย
ใช้ MediaPipe Pose ในการตรวจจับจุด landmark บนร่างกาย (33 จุด)
แล้วคำนวณมุมข้อต่อเพื่อนับจำนวนครั้ง (rep) และประเมินความถูกต้องของฟอร์ม

หากมีโมเดลที่เทรนเองจาก Colab (เช่น CNN/LSTM ตรวจสอบฟอร์มผิด-ถูก)
ให้โหลดในฟังก์ชัน load_custom_model() ด้านล่าง แล้วเรียกใช้ใน analyze_form()
"""
import cv2
import numpy as np
import mediapipe as mp
import math
import base64
from typing import Optional

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class PoseDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        # state ต่อ session การออกกำลังกาย (เก็บใน memory ฝั่ง backend ระหว่าง stream)
        self.counters = {}  # session_id -> {"count": int, "stage": str}

        self.custom_model = self.load_custom_model()

    # ---------- โหลดโมเดลที่เทรนจาก Colab (ถ้ามี) ----------
    def load_custom_model(self):
        """
        โหลดโมเดล .h5 / .tflite ที่เทรนจาก Colab (ดู colab/train_pose_classifier.ipynb)
        คาดหวัง input เป็น vector ของมุมข้อต่อ/landmark แล้ว output เป็นคลาส
        เช่น ['correct_squat', 'knee_too_far', 'back_not_straight', ...]
        """
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model("app/ml_models/exercise_form_classifier.h5")
            return model
        except Exception:
            # ถ้ายังไม่มีโมเดล ระบบจะ fallback ไปใช้กฎมุมข้อต่อ (rule-based) ด้านล่าง
            return None

    @staticmethod
    def calculate_angle(a, b, c) -> float:
        """คำนวณมุม (degree) ระหว่างจุด 3 จุด โดย b คือจุดยอด (vertex)"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = (
            math.atan2(c[1] - b[1], c[0] - b[0])
            - math.atan2(a[1] - b[1], a[0] - b[0])
        )
        angle = abs(radians * 180.0 / math.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def extract_landmarks(self, results, image_shape):
        """แปลง landmark ของ mediapipe เป็น dict {ชื่อจุด: (x, y)} ตามขนาดภาพจริง"""
        h, w = image_shape[:2]
        landmarks = {}
        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                name = mp_pose.PoseLandmark(idx).name
                landmarks[name] = (lm.x * w, lm.y * h, lm.visibility)
        return landmarks

    # ---------- ตรรกะนับจำนวนครั้งแยกตามชนิดท่า ----------
    def count_squat(self, landmarks, session_id):
        hip = landmarks.get("LEFT_HIP")
        knee = landmarks.get("LEFT_KNEE")
        ankle = landmarks.get("LEFT_ANKLE")
        if not all([hip, knee, ankle]):
            return None, "none", "ไม่พบร่างกายในเฟรม กรุณาขยับให้กล้องเห็นเต็มตัว"

        angle = self.calculate_angle(hip[:2], knee[:2], ankle[:2])
        state = self.counters.setdefault(session_id, {"count": 0, "stage": "up"})

        feedback = "ฟอร์มดี"
        if angle < 90:
            state["stage"] = "down"
            if angle < 60:
                feedback = "ลงลึกเกินไป ระวังเข่า"
        if angle > 160 and state["stage"] == "down":
            state["stage"] = "up"
            state["count"] += 1
            feedback = "นับครบ 1 ครั้ง!"
        elif angle <= 160 and angle >= 90:
            feedback = "งอเข่าต่อไปจนกว่ามุมสะโพกจะน้อยกว่า 90 องศา"

        return angle, state, feedback

    def count_pushup(self, landmarks, session_id):
        shoulder = landmarks.get("LEFT_SHOULDER")
        elbow = landmarks.get("LEFT_ELBOW")
        wrist = landmarks.get("LEFT_WRIST")
        if not all([shoulder, elbow, wrist]):
            return None, "none", "ไม่พบร่างกายในเฟรม"

        angle = self.calculate_angle(shoulder[:2], elbow[:2], wrist[:2])
        state = self.counters.setdefault(session_id, {"count": 0, "stage": "up"})

        feedback = "ฟอร์มดี"
        if angle < 90:
            state["stage"] = "down"
        if angle > 160 and state["stage"] == "down":
            state["stage"] = "up"
            state["count"] += 1
            feedback = "นับครบ 1 ครั้ง!"

        return angle, state, feedback

    def count_plank(self, landmarks, session_id):
        shoulder = landmarks.get("LEFT_SHOULDER")
        hip = landmarks.get("LEFT_HIP")
        ankle = landmarks.get("LEFT_ANKLE")
        if not all([shoulder, hip, ankle]):
            return None, "none", "ไม่พบร่างกายในเฟรม"

        angle = self.calculate_angle(shoulder[:2], hip[:2], ankle[:2])
        feedback = "ลำตัวตรงดี รักษาฟอร์มนี้ไว้" if 160 <= angle <= 180 else "สะโพกตกหรือยกสูงเกินไป ปรับให้ลำตัวเป็นเส้นตรง"
        state = self.counters.setdefault(session_id, {"count": 0, "stage": "hold"})
        return angle, state, feedback

    # ---------- ฟังก์ชันหลักที่ FastAPI endpoint จะเรียกใช้ ----------
    def analyze_frame(self, frame: np.ndarray, exercise_type: str, session_id: str) -> dict:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        landmarks = self.extract_landmarks(results, frame.shape)

        if not landmarks:
            return {
                "exercise_type": exercise_type,
                "rep_count": self.counters.get(session_id, {}).get("count", 0),
                "stage": "none",
                "form_feedback": "ไม่พบร่างกายในเฟรม กรุณาปรับตำแหน่งกล้อง",
                "accuracy_score": 0.0,
                "angle_data": {},
            }

        dispatch = {
            "squat": self.count_squat,
            "pushup": self.count_pushup,
            "plank": self.count_plank,
        }
        handler = dispatch.get(exercise_type, self.count_squat)
        angle, state, feedback = handler(landmarks, session_id)

        accuracy = self._estimate_accuracy(landmarks, exercise_type)
        stage_str = state["stage"] if isinstance(state, dict) else "none"
        count = state["count"] if isinstance(state, dict) else 0

        # วาดจุด landmark ลงบนภาพ (สำหรับส่งกลับไปแสดงผลฝั่ง Streamlit)
        annotated = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        return {
            "exercise_type": exercise_type,
            "rep_count": count,
            "stage": stage_str,
            "form_feedback": feedback,
            "accuracy_score": accuracy,
            "angle_data": {"primary_angle": round(angle, 1) if angle else 0},
            "annotated_frame_b64": self._encode_frame(annotated),
        }

    def _estimate_accuracy(self, landmarks, exercise_type) -> float:
        """
        ถ้ามี custom_model (เทรนจาก Colab) ให้ใช้โมเดลทำนายความถูกต้องของฟอร์ม
        ถ้าไม่มี ให้ประเมินคร่าวๆ จาก visibility ของจุดสำคัญ (rule-based fallback)
        """
        if self.custom_model is not None:
            try:
                vector = self._landmarks_to_vector(landmarks)
                pred = self.custom_model.predict(vector[np.newaxis, ...], verbose=0)
                return float(np.max(pred) * 100)
            except Exception:
                pass

        key_points = ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"]
        visible = [landmarks[p][2] for p in key_points if p in landmarks]
        if not visible:
            return 0.0
        return round(float(np.mean(visible) * 100), 1)

    @staticmethod
    def _landmarks_to_vector(landmarks) -> np.ndarray:
        order = [lm.name for lm in mp_pose.PoseLandmark]
        vec = []
        for name in order:
            x, y, v = landmarks.get(name, (0, 0, 0))
            vec.extend([x, y, v])
        return np.array(vec, dtype=np.float32)

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

    def reset_session(self, session_id: str):
        self.counters.pop(session_id, None)


# instance เดียวใช้ร่วมกันทั้งแอป (โหลดโมเดลครั้งเดียวตอน startup)
pose_detector = PoseDetector()
