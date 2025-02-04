import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import supervision as sv

# Initialize YOLO model
model = YOLO("yolo11s.pt")
names = model.model.names

# OpenCV VideoCapture (Use a video file or webcam)
cap = cv2.VideoCapture('vidp.mp4')

count = 0
boxCornerAnnotator = sv.BoxCornerAnnotator()
mask_annotator = sv.MaskAnnotator()  # MaskAnnotator instance

while True:
    
    # OpenCV: Read frame from video
    ret, frame = cap.read()
    if not ret:
        # Reset to the start of the video if the video ends
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue  # Process every third frame

    frame = cv2.resize(frame, (1020, 600))

    # YOLO: Run tracking on the frame
    results = model.track(frame, persist=True, classes=0)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get YOLO detections (bounding boxes, class IDs, track IDs)
        boxes = results[0].boxes.xyxy.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Filter out the person class (class_id == 0)
        person_count = 0
        for class_id in class_ids:
            if class_id == 0:  # Class ID 0 is "person" in YOLO
                person_count += 1

        # Annotate the frame with the count of people
        cv2.putText(frame, f"Persons detected: {person_count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Detections for MaskAnnotator (using bounding boxes as a placeholder)
        detections = sv.Detections(xyxy=boxes, class_id=np.array(class_ids), 
                                   tracker_id=np.array(track_ids))

        # Annotating with BoxCornerAnnotator
        annotatedFrame = boxCornerAnnotator.annotate(frame, detections)

        # Simulate masks using semi-transparent bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box
            mask = frame.copy()
            mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Simulate mask with green rectangle
            frame = cv2.addWeighted(mask, 0.5, frame, 0.5, 0)  # Blend the mask with the original frame

        # Annotating with MaskAnnotator (this works for segmentation masks, here using boxes as placeholders)
        annotatedFrame = mask_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

    # Show the original frame in OpenCV window with annotated bounding boxes and track ID
    cv2.imshow("RGB", annotatedFrame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
