import cv2
from ultralytics import YOLO

MODEL_PATH = '../production/assets/models/2025-04-09/best.pt'

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f'Error loading model: {e}')
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Error: could not open web cam.')
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

while True:
    succes, frame = cap.read()
    if not succes:
        print('Error: could not read frame.')
        break

    # Perform inference
    print(frame.shape)
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO Inference', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
