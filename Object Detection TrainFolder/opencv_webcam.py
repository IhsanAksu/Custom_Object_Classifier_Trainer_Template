import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/my_tool_model/weights/best.pt')

threshold = 0.07  
imagesize = 640

# Connect to the OBS Virtual Camera
# If 0 doesn't work, try 1 or 2
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(frame, imgsz=imagesize, conf=threshold, verbose=False)

    # Visualize the results
    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLO Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
