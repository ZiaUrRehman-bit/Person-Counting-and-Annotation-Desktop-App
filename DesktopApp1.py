import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import threading
from ultralytics import YOLO
import supervision as sv

class VideoAnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Annotator")
        self.root.geometry("400x200")
        
        self.video_path = None
        self.annotation_mode = "Mask"
        self.model = YOLO("yolo11s.pt")
        
        self.load_button = tk.Button(root, text="Load Video", command=self.load_video)
        self.load_button.pack(pady=5)
        
        self.mode_var = tk.StringVar(value="Mask")
        self.mask_button = tk.Radiobutton(root, text="Mask", variable=self.mode_var, value="Mask", command=self.set_mode)
        self.round_button = tk.Radiobutton(root, text="Round", variable=self.mode_var, value="Round", command=self.set_mode)
        self.triangle_button = tk.Radiobutton(root, text="Triangle", variable=self.mode_var, value="Triangle", command=self.set_mode)
        
        self.mask_button.pack()
        self.round_button.pack()
        self.triangle_button.pack()
        
        self.play_button = tk.Button(root, text="Play Video", command=self.start_video)
        self.play_button.pack(pady=5)
        
    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            print(f"Loaded video: {self.video_path}")
    
    def set_mode(self):
        self.annotation_mode = self.mode_var.get()
        print(f"Annotation mode set to: {self.annotation_mode}")
    
    def start_video(self):
        if not self.video_path:
            print("Please load a video first!")
            return
        
        thread = threading.Thread(target=self.process_video)
        thread.start()
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        annotator = self.get_annotator()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (1020, 600))
            results = self.model.track(frame, persist=True, classes=0)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                detections = sv.Detections(xyxy=boxes, class_id=np.array(class_ids), tracker_id=np.array(track_ids))
                frame = annotator.annotate(frame, detections)
            
            cv2.imshow("Video Annotator", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def get_annotator(self):
        if self.annotation_mode == "Mask":
            return sv.MaskAnnotator()
        elif self.annotation_mode == "Round":
            return sv.RoundBoxAnnotator()
        elif self.annotation_mode == "Triangle":
            return sv.TriangleAnnotator()
        return sv.BoxCornerAnnotator()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotatorApp(root)
    root.mainloop()
