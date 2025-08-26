import cv2
from ultralytics import YOLO
import mediapipe as mp

class ObjectDetector:
    def __init__(self, yolo_model_path="yolov8n.pt", use_gpu=False):
        # Load YOLO model
        device = "cpu"
        self.yolo_model = YOLO(yolo_model_path, device=device)

        # Initialize MediaPipe for faces
        self.mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, max_num_hands=2)

    def detect(self, frame):
        detections = []

        # --- YOLO Detection for custom objects ---
        results = self.yolo_model(frame, stream=True)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.yolo_model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                detections.append({
                    "type": label,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        # --- MediaPipe Face Detection ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.mp_face.process(rgb_frame)
        if face_results.detections:
            for face in face_results.detections:
                bboxC = face.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                detections.append({
                    "type": "face",
                    "confidence": face.score[0],
                    "bbox": (x1, y1, x1 + w, y1 + h)
                })

        # --- MediaPipe Hands Detection ---
        hand_results = self.mp_hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                ih, iw, _ = frame.shape
                x1, x2 = int(min(x_coords) * iw), int(max(x_coords) * iw)
                y1, y2 = int(min(y_coords) * ih), int(max(y_coords) * ih)
                detections.append({
                    "type": "hand",
                    "confidence": 1.0,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections
