from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Takes a single frame (numpy array) and returns bounding boxes
        Format: list of dicts: {'xyxy': [x1, y1, x2, y2], 'conf': 0.9, 'class': 'middle'}
        """
        results = self.model(frame)  # inference
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append({
                    'xyxy': [x1, y1, x2, y2],
                    'conf': conf,
                    'class': self.model.names[cls_id]
                })
        return detections
