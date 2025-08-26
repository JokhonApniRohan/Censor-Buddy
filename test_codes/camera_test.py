import cv2
from video_capture import VideoCapture

cap = VideoCapture(camera_index=0, width=640, height=480)

while True:
    frame = cap.read()
    if frame is None:
        print("❌ Could not read frame. Exiting...")
        break

    # Show the video frame
    cv2.imshow("Webcam Test", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
