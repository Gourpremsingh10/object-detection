from ultralytics import YOLO
import cv2

def main():
    # Load a small, fast pretrained model (auto-downloads ~6MB first run)
    model = YOLO("yolo11n.pt")   # alternatives: yolov8n.pt, yolo26n.pt

    # Open webcam (0 = default camera). Use "video.mp4" for a file instead.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try a different index (1, 2).")

    print("Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on the frame
        results = model(frame, verbose=False)

        # Draw boxes + labels on the frame
        annotated = results[0].plot()

        cv2.imshow("YOLO Object Detection - press q to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()