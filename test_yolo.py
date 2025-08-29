import cv2
from video_capture import VideoCapture
from object_detection import ObjectDetector

# Initialize webcam and YOLO detector
cap = VideoCapture(2)  # working camera index
detector = ObjectDetector("runs/detect/train/weights/best.pt")

while True:
    frame = cap.read()
    if frame is None:
        break

    # Detect objects
    detections = detector.detect(frame)

    # Blur detected objects
    for det in detections:
        x1, y1, x2, y2 = det['xyxy']
        roi = frame[y1:y2, x1:x2]
        roi = cv2.GaussianBlur(roi, (81, 81), 0)
        frame[y1:y2, x1:x2] = roi

    # Show the frame
    cv2.imshow("Censored Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
