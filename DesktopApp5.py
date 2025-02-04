import tkinter as tk
from tkinter import filedialog, Label, Frame, Canvas, ttk, messagebox
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
        self.root.geometry("1200x750")  # Increased window size
        
        self.video_path = None
        self.annotation_mode = "Ellips"
        self.model = YOLO("yolo11s.pt")
        self.running = False  # Flag to control video playback
        
        # Layout setup
        self.control_frame = Frame(root, width=300, height=750, bg="#2C3E50")  # Dark grayish-blue
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.video_frame = Frame(root, width=900, height=750, bg="#34495E")
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = Canvas(self.video_frame, width=900, height=750, bg="black")
        self.canvas.pack()
        
        # Controls
        self.load_button = tk.Button(self.control_frame, text="Load Video", command=self.load_video, bg="#1ABC9C", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT)
        self.load_button.pack(pady=10, padx=10, fill=tk.X)
        
        # Dropdown Menu for Annotation Selection
        self.mode_var = tk.StringVar(value="Ellips")
        self.mode_label = Label(self.control_frame, text="Select Annotation Mode", bg="#2C3E50", fg="white", font=("Arial", 12, "bold"))
        self.mode_label.pack(pady=5)
        
        self.mode_dropdown = ttk.Combobox(self.control_frame, textvariable=self.mode_var, 
                                          values=["Ellips", "RoundBox", "Triangle", "HeatMap", "Label", "Trace", "Pixelate", "BoxCorner", "Circle", "Blur"], 
                                          state="readonly", font=("Arial", 12))
        self.mode_dropdown.pack(pady=5, padx=10, fill=tk.X)
        self.mode_dropdown.bind("<<ComboboxSelected>>", lambda event: self.set_mode())
        
        self.play_button = tk.Button(self.control_frame, text="Play Video", command=self.start_video, bg="#E74C3C", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT)
        self.play_button.pack(pady=20, padx=10, fill=tk.X)
        
        self.info_frame = Frame(self.control_frame, bg="#2C3E50", pady=20)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.info_label = Label(self.info_frame, text="Persons Detected: 0", bg="#1F618D", fg="white", font=("Arial", 14, "bold"), relief=tk.RIDGE, padx=10, pady=5)
        self.info_label.pack(pady=5, padx=10, fill=tk.X)
        
        self.mode_info_label = Label(self.info_frame, text=f"Mode: {self.annotation_mode}", bg="#1F618D", fg="white", font=("Arial", 14, "bold"), relief=tk.RIDGE, padx=10, pady=5)
        self.mode_info_label.pack(pady=5, padx=10, fill=tk.X)
        
    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            messagebox.showinfo("Video Loaded", f"Successfully loaded video: {self.video_path}")
            # print(f"Loaded video: {self.video_path}")
    
    def set_mode(self):
        self.annotation_mode = self.mode_var.get()
        # print(f"Annotation mode set to: {self.annotation_mode}")
        self.mode_info_label.config(text=f"Mode: {self.annotation_mode}")
        self.running = False  # Stop current video processing
        
    def start_video(self):
        if not self.video_path:
            # print("Please load a video first!")
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
            
            frame = cv2.resize(frame, (900, 750))
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
        elif self.annotation_mode == "RoundBox":
            return sv.RoundBoxAnnotator()
        elif self.annotation_mode == "Triangle":
            return sv.TriangleAnnotator()
        elif self.annotation_mode == "HeatMap":
            return sv.HeatMapAnnotator()
        elif self.annotation_mode == "Label":
            return sv.LabelAnnotator()
        elif self.annotation_mode == "Trace":
            return sv.TraceAnnotator()
        elif self.annotation_mode == "Pixelate":
            return sv.PixelateAnnotator()
        elif self.annotation_mode == "BoxCorner":
            return sv.BoxCornerAnnotator()
        elif self.annotation_mode == "Blur":
            return sv.BlurAnnotator()
        elif self.annotation_mode == "Circle":
            return sv.CircleAnnotator()
        return sv.BoxCornerAnnotator()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotatorApp(root)
    root.mainloop()