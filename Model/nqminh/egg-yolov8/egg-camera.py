from ultralytics import YOLO
import cv2

model = YOLO("/media/qminh/New Volume/qm/USTH/COURSES/B3/Introduction to Deeplearning/Dataset/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Egg Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()
