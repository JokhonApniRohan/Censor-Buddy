from ultralytics import YOLO
import cv2

def test_yolo():
    # Load YOLOv8 pretrained model (small & fast)
    model = YOLO("yolov8n.pt")  # You can also try yolov8s.pt for better accuracy

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Run YOLO object detection
        results = model(frame, stream=True)

        # Draw detections on the frame
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
                conf = box.conf[0]            # Confidence score
                cls = int(box.cls[0])         # Class ID
                label = model.names[cls]      # Class label

                # Draw rectangle + label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

        # Show output
        cv2.imshow("YOLOv8 Live Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_yolo()
