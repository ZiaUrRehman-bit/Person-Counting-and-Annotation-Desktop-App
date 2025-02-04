import tkinter as tk
from tkinter import filedialog, Label, Frame, Canvas
import cv2
import numpy as np
import threading
from ultralytics import YOLO
import supervision as sv
from PIL import Image, ImageTk

class VideoAnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Annotator")
        self.root.geometry("1000x700")  # Increased window size
        
        self.video_path = None
        self.annotation_mode = "Mask"
        self.model = YOLO("yolo11s.pt")
        self.running = False  # Flag to control video playback
        
        # Layout setup
        self.control_frame = Frame(root, width=250, height=700, bg="#2C3E50")  # Dark grayish-blue
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.video_frame = Frame(root, width=750, height=700)
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = Canvas(self.video_frame, width=750, height=700, bg="black")
        self.canvas.pack()
        
        # Controls
        self.load_button = tk.Button(self.control_frame, text="Load Video", command=self.load_video, bg="#1ABC9C", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT)
        self.load_button.pack(pady=10, padx=10, fill=tk.X)
        
        self.mode_var = tk.StringVar(value="Ellips")
        self.mask_button = tk.Radiobutton(self.control_frame, text="Ellips", variable=self.mode_var, value="Ellips", command=self.set_mode, bg="#2C3E50", fg="white", font=("Arial", 12, "bold"))
        self.round_button = tk.Radiobutton(self.control_frame, text="Round", variable=self.mode_var, value="Round", command=self.set_mode, bg="#2C3E50", fg="white", font=("Arial", 12, "bold"))
        self.triangle_button = tk.Radiobutton(self.control_frame, text="Triangle", variable=self.mode_var, value="Triangle", command=self.set_mode, bg="#2C3E50", fg="white", font=("Arial", 12, "bold"))
        
        self.mask_button.pack(pady=5, padx=10, anchor=tk.W)
        self.round_button.pack(pady=5, padx=10, anchor=tk.W)
        self.triangle_button.pack(pady=5, padx=10, anchor=tk.W)
        
        self.play_button = tk.Button(self.control_frame, text="Play Video", command=self.start_video, bg="#E74C3C", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT)
        self.play_button.pack(pady=20, padx=10, fill=tk.X)
        
        self.info_label = Label(self.control_frame, text="Persons Detected: 0", bg="#2C3E50", fg="white", font=("Arial", 12, "bold"))
        self.info_label.pack(pady=10)
        
    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            print(f"Loaded video: {self.video_path}")
    
    def set_mode(self):
        self.annotation_mode = self.mode_var.get()
        print(f"Annotation mode set to: {self.annotation_mode}")
        self.running = False  # Stop current video processing
        
    def start_video(self):
        if not self.video_path:
            print("Please load a video first!")
            return
        
        self.running = False  # Stop any ongoing processing before switching mode
        thread = threading.Thread(target=self.process_video, daemon=True)
        thread.start()
    
    def process_video(self):
        self.running = True
        cap = cv2.VideoCapture(self.video_path)
        annotator = self.get_annotator()
        count = 0

        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count % 3 != 0:
                continue  # Process every third frame
            
            frame = cv2.resize(frame, (750, 700))
            results = self.model.track(frame, persist=True, classes=0)
            
            person_count = 0
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                detections = sv.Detections(xyxy=boxes, class_id=np.array(class_ids), tracker_id=np.array(track_ids))
                frame = annotator.annotate(frame, detections)
                person_count = class_ids.count(0)
            
            self.root.after(0, self.display_frame, frame)
            self.root.after(0, self.update_info_label, person_count)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk
    
    def update_info_label(self, count):
        self.info_label.config(text=f"Persons Detected: {count}")
    
    def get_annotator(self):
        if self.annotation_mode == "Ellips":
            return sv.EllipseAnnotator()
        elif self.annotation_mode == "Round":
            return sv.RoundBoxAnnotator()
        elif self.annotation_mode == "Triangle":
            return sv.TriangleAnnotator()
        return sv.BoxCornerAnnotator()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotatorApp(root)
    root.mainloop()
