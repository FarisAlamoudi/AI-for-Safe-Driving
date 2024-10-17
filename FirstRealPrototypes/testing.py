from ultralytics import YOLO
import cv2
import time

model = YOLO('/home/jbeni/Desktop/School/COP4934/AI-for-Safe-Driving/prototyping/runs/detect/train20/weights/best.onnx')
cap = cv2.VideoCapture(0)
start = time.time()
count = 0

while True:
    count += 1 
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    cv2.imshow('YOLO Video Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end = time.time()
print(f'Time elapsed: {end - start}s\nRunning at {count/(end-start)}fps')