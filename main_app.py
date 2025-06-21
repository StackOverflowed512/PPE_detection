import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import datetime
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import json
from ultralytics import YOLO
import sqlite3
import shutil # For copying files
import pickle # For serializing/deserializing numpy arrays for DB

DATABASE_NAME = "personnel_data.db"
REGISTERED_IMAGES_DIR = "registered_images"


CONFIG = {}
# These will be populated from the database
db_known_face_encodings = []
db_known_face_metadata_list = [] # List of dicts, index-matched with encodings
db_display_name_to_metadata_map = {} # For quick lookup by display_name for info panel

mouse_x, mouse_y = 0, 0
person_to_show_details = None
detected_face_boxes_for_hover = []

yolo_ppe_model = None
ppe_class_names_from_model = {}
ppe_class_colors_from_config = {}

# --- Database Functions ---
def init_db():
    if not os.path.exists(REGISTERED_IMAGES_DIR):
        os.makedirs(REGISTERED_IMAGES_DIR)
        print(f"Created directory: {REGISTERED_IMAGES_DIR}")

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        display_name TEXT NOT NULL,
        age INTEGER,
        function_text TEXT,
        person_id_code TEXT UNIQUE, 
        hashcode TEXT,
        image_filename TEXT NOT NULL,
        face_encoding BLOB NOT NULL
    )
    """)
    conn.commit()
    conn.close()
    print("Database initialized.")

def add_person_to_db(display_name, age, function_text, person_id_code, hashcode, image_filename, face_encoding):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    try:
        # Serialize numpy array (face_encoding) to bytes using pickle
        serialized_encoding = pickle.dumps(face_encoding)
        cursor.execute("""
        INSERT INTO persons (display_name, age, function_text, person_id_code, hashcode, image_filename, face_encoding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (display_name, age, function_text, person_id_code, hashcode, image_filename, serialized_encoding))
        conn.commit()
        print(f"Successfully added {display_name} to the database.")
    except sqlite3.IntegrityError:
        print(f"ERROR: Person ID Code '{person_id_code}' already exists in the database. Registration failed.")
    except Exception as e:
        print(f"ERROR: Could not add person to database: {e}")
    finally:
        conn.close()

def load_persons_from_db():
    global db_known_face_encodings, db_known_face_metadata_list, db_display_name_to_metadata_map
    db_known_face_encodings.clear()
    db_known_face_metadata_list.clear()
    db_display_name_to_metadata_map.clear()

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT display_name, age, function_text, person_id_code, hashcode, image_filename, face_encoding FROM persons")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No persons found in the database.")
        return

    for row in rows:
        display_name, age, function_text, person_id_code, hashcode, image_filename, serialized_encoding = row
        
        try:
            # Deserialize bytes back to numpy array
            face_encoding = pickle.loads(serialized_encoding)
            db_known_face_encodings.append(face_encoding)
            
            metadata = {
                "display_name": display_name,
                "AGE": age, # Keep JSON key style for compatibility with drawing function
                "Function": function_text, # Keep JSON key style
                "ID": person_id_code, # Keep JSON key style
                "HASHCODE": hashcode, # Keep JSON key style
                "image_filename": image_filename,
                "image_path_for_panel": os.path.join(REGISTERED_IMAGES_DIR, image_filename)
            }
            db_known_face_metadata_list.append(metadata)
            db_display_name_to_metadata_map[display_name] = metadata
        except Exception as e:
            print(f"Error loading person {display_name} from DB (encoding/metadata issue): {e}")
            
    print(f"Loaded {len(db_known_face_encodings)} known faces from the database.")

# --- Registration Function ---
def register_new_person_cli():
    print("\n--- New Person Registration ---")
    display_name = input("Enter display name: ")
    try:
        age = int(input("Enter age: "))
    except ValueError:
        print("Invalid age. Setting to N/A.")
        age = None # Or handle as an error
    function_text = input("Enter function/role: ")
    person_id_code = input("Enter unique Person ID Code (e.g., employee ID): ")
    hashcode = input("Enter HASHCODE (optional, press Enter to skip): ")
    image_path_input = input("Enter full path to the person's image file: ")

    if not os.path.exists(image_path_input):
        print(f"ERROR: Image file not found at '{image_path_input}'. Registration aborted.")
        return

    try:
        image = face_recognition.load_image_file(image_path_input)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            print(f"ERROR: No face found in the image '{image_path_input}'. Registration aborted.")
            return
        
        face_encoding = encodings[0] # Use the first face found

        # Create a unique filename for storage (e.g., using timestamp or name)
        base_filename = os.path.basename(image_path_input)
        name_part, ext_part = os.path.splitext(base_filename)
        # Sanitize display_name for filename
        sanitized_name = "".join(c if c.isalnum() else "_" for c in display_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        new_image_filename = f"{sanitized_name}_{person_id_code}_{timestamp}{ext_part}"
        
        destination_path = os.path.join(REGISTERED_IMAGES_DIR, new_image_filename)
        shutil.copy(image_path_input, destination_path)
        print(f"Image copied to {destination_path}")

        add_person_to_db(display_name, age, function_text, person_id_code, hashcode, new_image_filename, face_encoding)

    except Exception as e:
        print(f"An error occurred during registration: {e}")

# --- Configuration and Data Loading (Modified) ---
def load_config(config_path="config.json"):
    global CONFIG, ppe_class_colors_from_config
    try:
        with open(config_path, 'r') as f:
            CONFIG = json.load(f)
        ppe_class_colors_from_config = CONFIG.get("ppe_class_colors", {"default": [100,100,100]})
        print("Configuration loaded.")
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{config_path}' not found. Exiting.")
        exit()
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from '{config_path}'. Check its format. Exiting.")
        exit()

# This function is replaced by load_persons_from_db()
# def load_known_persons_data(data_path="known_persons_data.json"): ...

# This function is replaced by load_persons_from_db()
# def initialize_known_faces(known_faces_dir="known_faces"): ...

def initialize_ppe_model():
    global yolo_ppe_model, ppe_class_names_from_model
    model_path = CONFIG.get("yolov8_ppe_model_path")
    if not model_path:
        print("YOLOv8 PPE model path not specified in config. PPE detection will be skipped.")
        return
    if not os.path.exists(model_path):
        print(f"YOLOv8 PPE model file not found at '{model_path}'. PPE detection will be skipped.")
        return
    try:
        yolo_ppe_model = YOLO(model_path)
        ppe_class_names_from_model = yolo_ppe_model.names
        print(f"YOLOv8 PPE detection model loaded successfully from '{model_path}'.")
        print(f"PPE Classes from model: {ppe_class_names_from_model}")
    except Exception as e:
        print(f"Error loading YOLOv8 PPE model: {e}")
        yolo_ppe_model = None

def detect_ppe_yolov8(frame):
    if yolo_ppe_model is None:
        return [], {}

    ppe_detections = []
    results = yolo_ppe_model.predict(source=frame, verbose=False, conf=CONFIG.get("ppe_confidence_threshold", 0.25))
    
    if results and results[0]:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = ppe_class_names_from_model.get(class_id, f"ClassID_{class_id}")
            color = ppe_class_colors_from_config.get(label, ppe_class_colors_from_config.get("default", [100,100,100]))
            ppe_detections.append((label, confidence, color, [x1, y1, x2 - x1, y2 - y1]))
    return ppe_detections, {}

# --- Drawing Functions (Mostly Unchanged) ---
def draw_info_panel(frame, metadata):
    if not metadata: return
    panel_x, panel_y, panel_w, panel_h = 350, 50, 380, 200
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 1)

    text_color = (255, 255, 255); line_height = 22; current_y = panel_y + 25
    font_scale = CONFIG.get("info_panel_font_scale", 0.6)

    cv2.putText(frame, "{Match}", (panel_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1); current_y += line_height
    cv2.putText(frame, f"Name: {metadata.get('display_name', 'N/A')}", (panel_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1); current_y += line_height
    cv2.putText(frame, f"AGE: {metadata.get('AGE', 'N/A')}", (panel_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1); current_y += line_height
    cv2.putText(frame, f"Function: {metadata.get('Function', 'N/A')}", (panel_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1); current_y += line_height
    cv2.putText(frame, f"ID: {metadata.get('ID', 'N/A')}", (panel_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1); current_y += line_height
    cv2.putText(frame, f"HASHCODE: {metadata.get('HASHCODE', 'N/A')}", (panel_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
    
    img_path = metadata.get("image_path_for_panel") # This path is now constructed from REGISTERED_IMAGES_DIR
    if img_path and os.path.exists(img_path):
        profile_pic = cv2.imread(img_path)
        if profile_pic is not None:
            pic_h, pic_w = 80, 80
            profile_pic_resized = cv2.resize(profile_pic, (pic_w, pic_h))
            pic_x_start = panel_x + panel_w - pic_w - 10; pic_y_start = panel_y + 10
            try: frame[pic_y_start:pic_y_start+pic_h, pic_x_start:pic_x_start+pic_w] = profile_pic_resized
            except ValueError: print("Warning: Profile picture placement error (check panel/image dimensions).")
    cv2.putText(frame, "Match 100%", (panel_x + panel_w - 130, panel_y + panel_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)

def draw_status_bar(frame, frame_width):
    bar_height = 60; bar_y_start = frame.shape[0] - bar_height
    overlay = frame.copy(); cv2.rectangle(overlay, (0, bar_y_start), (frame_width, frame.shape[0]), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (0, bar_y_start), (frame_width, frame.shape[0]), (0, 255, 255), 2)
    text_color = (255, 255, 255); on_color = (0, 255, 0); font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = CONFIG.get("status_bar_font_scale", 0.7); thickness = 1
    y_pos = bar_y_start + int(bar_height / 2) + 5
    texts = [("Personnel monitoring:", "ON"), ("Facial recognition:", "ON"), ("Object tracking:", "ON"), ("Heatmap:", "ON")]
    current_x = 20
    for label, status in texts:
        cv2.putText(frame, label, (current_x, y_pos), font, font_scale, text_color, thickness)
        text_size_label = cv2.getTextSize(label, font, font_scale, thickness)[0]; current_x += text_size_label[0] + 5
        cv2.putText(frame, status, (current_x, y_pos), font, font_scale, on_color, thickness, cv2.LINE_AA)
        text_size_status = cv2.getTextSize(status, font, font_scale, thickness)[0]; current_x += text_size_status[0] + 30

def draw_hud_elements(frame, frame_width):
    camera_name = CONFIG.get("camera_name", "Camera")
    cv2.rectangle(frame, (10, 10), (len(camera_name)*15 + 40, 50), (0,0,0), cv2.FILLED)
    cv2.putText(frame, camera_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.rectangle(frame, (frame_width - 170, 10), (frame_width - 10, 50), (0,0,0), cv2.FILLED)
    cv2.putText(frame, time_str, (frame_width - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

def mouse_event_handler(event, x, y, flags, param):
    global mouse_x, mouse_y, person_to_show_details
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
        found_hover_person_name = None # Store name of the person hovered on
        if CONFIG.get("display_info_panel_on_hover", False):
            for name, box in detected_face_boxes_for_hover:
                left, top, right, bottom = box
                if left < mouse_x < right and top < mouse_y < bottom:
                    found_hover_person_name = name
                    break
            person_to_show_details = found_hover_person_name # This is the display_name

# --- Main Application Logic ---
def main():
    global detected_face_boxes_for_hover, person_to_show_details

    init_db() # Initialize database and tables
    load_config() # Load general app configuration
    
    action = input("Choose action: (R)egister new person or (D)etect: ").strip().upper()
    if action == 'R':
        root = tk.Tk()
        app = RegistrationForm(root)
        root.mainloop()
        return
    elif action != 'D':
        print("Invalid choice. Starting detection by default.")

    load_persons_from_db() # Load faces and metadata from the database
    initialize_ppe_model()

    cap_source = CONFIG.get("camera_id", 0)
    if isinstance(cap_source, str) and not cap_source.isdigit(): pass
    else: cap_source = int(cap_source)

    video_capture = cv2.VideoCapture(cap_source)
    if not video_capture.isOpened():
        print(f"Error: Could not open video source '{cap_source}'. Exiting."); return

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    window_name = CONFIG.get("output_window_title", "Video")
    cv2.namedWindow(window_name)
    if CONFIG.get("display_info_panel_on_hover", False):
        cv2.setMouseCallback(window_name, mouse_event_handler)

    unknown_id_counter = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            if isinstance(cap_source, str): print("End of video file."); break
            print("Error: Failed to capture frame."); break
        
        detected_face_boxes_for_hover.clear()
        if not CONFIG.get("display_info_panel_on_hover", False):
            person_to_show_details = None # Reset for current frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings_current_frame = face_recognition.face_encodings(rgb_frame, face_locations)

        ppe_detections_list, _ = detect_ppe_yolov8(frame)
        recognized_this_frame_details = []

        for i, current_face_encoding in enumerate(face_encodings_current_frame):
            name = "Unknown"
            person_id_display = "N/A"
            metadata_for_this_person = None
            
            if db_known_face_encodings: # Check if there are any known faces loaded
                matches = face_recognition.compare_faces(db_known_face_encodings, current_face_encoding, tolerance=CONFIG.get("face_recognition_tolerance", 0.6))
                face_distances = face_recognition.face_distance(db_known_face_encodings, current_face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        # Get metadata from the list using the best_match_index
                        metadata_for_this_person = db_known_face_metadata_list[best_match_index]
                        name = metadata_for_this_person.get('display_name', "ErrorName")
                        person_id_display = metadata_for_this_person.get("ID", "N/A") # 'ID' is the key used in draw_info_panel

            if name == "Unknown":
                if CONFIG.get("show_unknown_person_id", True):
                    person_id_display = f"UNKN_{unknown_id_counter}"
                else:
                    person_id_display = "Unknown"
                if CONFIG.get("log_unknown_detections", False):
                    print(f"Log: Unknown person detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            top, right, bottom, left = face_locations[i]
            if name != "Unknown" and metadata_for_this_person: # Ensure metadata exists
                 detected_face_boxes_for_hover.append((name, (left, top, right, bottom)))

            id_text_y_offset = -10
            id_text_content = f"ID {person_id_display}"
            (id_text_w, id_text_h), _ = cv2.getTextSize(id_text_content, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            id_bg_top_left = (left, top + id_text_y_offset - id_text_h - 5)
            id_bg_bottom_right = (left + id_text_w + 10, top + id_text_y_offset + 5)
            if id_bg_top_left[1] < 0:
                id_text_y_offset = bottom + 10 
                id_bg_top_left = (left, top + id_text_y_offset)
                id_bg_bottom_right = (left + id_text_w + 10, top + id_text_y_offset + id_text_h + 5)

            cv2.rectangle(frame, id_bg_top_left, id_bg_bottom_right, (0,0,0), cv2.FILLED)
            cv2.putText(frame, id_text_content, (left + 5, top + id_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 3)

            recognized_this_frame_details.append({
                "name": name, "id": person_id_display, "box": (left, top, right, bottom),
                "metadata": metadata_for_this_person
            })
        
        if any(p['name'] == "Unknown" for p in recognized_this_frame_details):
            if CONFIG.get("show_unknown_person_id", True): # only increment if we were actually showing UNKN IDs
                unknown_id_counter +=1

        for ppe_label, ppe_confidence, ppe_color, (px, py, pw, ph) in ppe_detections_list:
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), ppe_color, 2)
            label_text = f"{ppe_label}: {ppe_confidence:.2f}"
            text_y = py - 7 if py - 7 > 7 else py + 15
            cv2.putText(frame, label_text, (px + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ppe_color, 1, cv2.LINE_AA)

        unknown_face_count = sum(1 for p_detail in recognized_this_frame_details if p_detail["name"] == "Unknown")
        if unknown_face_count > 0:
            cv2.putText(frame, f"WARNING: {unknown_face_count} UNKNOWN FACE(S)", (frame_width // 2 - 200, 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)

        if person_to_show_details and person_to_show_details in db_display_name_to_metadata_map:
            draw_info_panel(frame, db_display_name_to_metadata_map[person_to_show_details])
        
        draw_hud_elements(frame, frame_width)
        draw_status_bar(frame, frame_width)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    video_capture.release()
    cv2.destroyAllWindows()

def update_person_in_db(person_id_code, display_name, age, function_text, hashcode):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE persons
            SET display_name=?, age=?, function_text=?, hashcode=?
            WHERE person_id_code=?
        """, (display_name, age, function_text, hashcode, person_id_code))
        conn.commit()
        print(f"Updated person {person_id_code} in the database.")
    except Exception as e:
        print(f"ERROR: Could not update person: {e}")
    finally:
        conn.close()

class EditPersonForm:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Edit Person")
        self.window.geometry("400x400")
        self.create_widgets()
        self.load_persons()

    def create_widgets(self):
        ttk.Label(self.window, text="Select Person to Edit:").pack(pady=10)
        self.person_var = tk.StringVar()
        self.person_combo = ttk.Combobox(self.window, textvariable=self.person_var, state="readonly")
        self.person_combo.pack(pady=5)
        self.person_combo.bind("<<ComboboxSelected>>", self.load_selected_person)

        self.fields = {}
        for label in ["Display Name", "Age", "Function", "Hashcode"]:
            ttk.Label(self.window, text=label + ":").pack()
            entry = ttk.Entry(self.window, width=40)
            entry.pack()
            self.fields[label] = entry

        ttk.Button(self.window, text="Update", command=self.update_person).pack(pady=20)

    def load_persons(self):
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT person_id_code, display_name FROM persons")
        self.persons = cursor.fetchall()
        conn.close()
        self.person_combo['values'] = [f"{pid} - {name}" for pid, name in self.persons]

    def load_selected_person(self, event=None):
        idx = self.person_combo.current()
        if idx < 0: return
        person_id_code = self.persons[idx][0]
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT display_name, age, function_text, hashcode FROM persons WHERE person_id_code=?", (person_id_code,))
        row = cursor.fetchone()
        conn.close()
        if row:
            for val, key in zip(row, ["Display Name", "Age", "Function", "Hashcode"]):
                self.fields[key].delete(0, tk.END)
                self.fields[key].insert(0, str(val) if val is not None else "")

    def update_person(self):
        idx = self.person_combo.current()
        if idx < 0:
            messagebox.showerror("Error", "Select a person to edit.")
            return
        person_id_code = self.persons[idx][0]
        display_name = self.fields["Display Name"].get()
        age = self.fields["Age"].get()
        function_text = self.fields["Function"].get()
        hashcode = self.fields["Hashcode"].get()
        try:
            update_person_in_db(person_id_code, display_name, int(age) if age else None, function_text, hashcode)
            load_persons_from_db()  # Refresh in-memory data
            messagebox.showinfo("Success", "Person updated successfully!")
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Update failed: {str(e)}")

class PersonRegistrationForm:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Person Registration")
        self.window.geometry("500x600")
        self.create_widgets()
        
    def create_widgets(self):
        # Create form fields with labels and entries
        ttk.Label(self.window, text="Person Registration", font=('Helvetica', 16, 'bold')).pack(pady=20)
        
        form_frame = ttk.Frame(self.window)
        form_frame.pack(padx=20, pady=10, fill='x')
        
        # Display Name
        ttk.Label(form_frame, text="Display Name:").pack(fill='x', pady=5)
        self.name_entry = ttk.Entry(form_frame, width=40)
        self.name_entry.pack(fill='x', pady=5)
        
        # Age
        ttk.Label(form_frame, text="Age:").pack(fill='x', pady=5)
        self.age_entry = ttk.Entry(form_frame, width=40)
        self.age_entry.pack(fill='x', pady=5)
        
        # Function/Role
        ttk.Label(form_frame, text="Function/Role:").pack(fill='x', pady=5)
        self.function_entry = ttk.Entry(form_frame, width=40)
        self.function_entry.pack(fill='x', pady=5)
        
        # Person ID
        ttk.Label(form_frame, text="Person ID Code:").pack(fill='x', pady=5)
        self.id_entry = ttk.Entry(form_frame, width=40)
        self.id_entry.pack(fill='x', pady=5)
        
        # Hashcode
        ttk.Label(form_frame, text="Hashcode (Optional):").pack(fill='x', pady=5)
        self.hashcode_entry = ttk.Entry(form_frame, width=40)
        self.hashcode_entry.pack(fill='x', pady=5)
        
        # Image selection
        ttk.Label(form_frame, text="Profile Image:").pack(fill='x', pady=5)
        self.image_path = tk.StringVar()
        ttk.Label(form_frame, textvariable=self.image_path).pack(fill='x', pady=5)
        ttk.Button(form_frame, text="Select Image", command=self.select_image).pack(fill='x', pady=10)
        
        # Submit button
        ttk.Button(form_frame, text="Register Person", command=self.register_person).pack(fill='x', pady=20)
        
    def select_image(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(title="Select Profile Image", filetypes=filetypes)
        if filename:
            self.image_path.set(filename)
            
    def register_person(self):
        # Gather form data
        data = {
            'display_name': self.name_entry.get(),
            'age': self.age_entry.get(),
            'function': self.function_entry.get(),
            'person_id': self.id_entry.get(),
            'hashcode': self.hashcode_entry.get(),
            'image_path': self.image_path.get()
        }
        
        # Validate required fields
        if not all([data['display_name'], data['age'], data['function'], 
                   data['person_id'], data['image_path']]):
            messagebox.showerror("Error", "Please fill all required fields")
            return
            
        try:
            # Call the registration function from main_app.py
            add_person_to_db(
                data['display_name'],
                int(data['age']),
                data['function'],
                data['person_id'],
                data['hashcode'],
                os.path.basename(data['image_path']),
                face_recognition.face_encodings(face_recognition.load_image_file(data['image_path']))[0]
            )
            messagebox.showinfo("Success", "Person registered successfully!")
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Registration failed: {str(e)}")

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Personnel Detection System")
        self.root.geometry("800x600")
        self.create_widgets()
        
    def create_widgets(self):
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Personnel Detection System",
            font=('Helvetica', 24, 'bold')
        )
        title_label.pack(pady=40)
        
        # Buttons container
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=40)
        
        # Register button
        register_btn = ttk.Button(
            button_frame,
            text="Register New Person",
            command=self.open_registration,
            width=20
        )
        register_btn.pack(pady=10)

        # Edit button
        edit_btn = ttk.Button(
            button_frame,
            text="Edit Person",
            command=self.open_edit_person,
            width=20
        )
        edit_btn.pack(pady=10)

        # Detect button
        detect_btn = ttk.Button(
            button_frame,
            text="Start Detection",
            command=self.start_detection,
            width=20
        )
        detect_btn.pack(pady=10)
        
    def open_registration(self):
        PersonRegistrationForm(self.root)
        
    def open_edit_person(self):
        EditPersonForm(self.root)
        
    def start_detection(self):
        self.root.withdraw()  # Hide the main window
        # Start the detection process
        try:
            init_db()
            load_config()
            load_persons_from_db()
            initialize_ppe_model()
            self.run_detection()
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
        finally:
            self.root.deiconify()  # Show the main window again
            
    def run_detection(self):
        # Add global variable references
        global detected_face_boxes_for_hover, person_to_show_details, unknown_id_counter
        
        # Initialize variables that were missing
        person_to_show_details = None
        unknown_id_counter = 0
        frame_width = 0

        cap_source = CONFIG.get("camera_id", 0)
        if isinstance(cap_source, str) and not cap_source.isdigit():
            cap_source = cap_source
        else:
            cap_source = int(cap_source)

        video_capture = cv2.VideoCapture(cap_source)
        if not video_capture.isOpened():
            messagebox.showerror("Error", "Could not open video source")
            return

        window_name = CONFIG.get("output_window_title", "Video")
        cv2.namedWindow(window_name)
        
        # Get frame width after opening video capture
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Set mouse callback for hover functionality
        if CONFIG.get("display_info_panel_on_hover", False):
            cv2.setMouseCallback(window_name, mouse_event_handler)

        # Run the existing detection loop
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    if isinstance(cap_source, str): print("End of video file."); break
                    print("Error: Failed to capture frame."); break
                
                detected_face_boxes_for_hover.clear()
                if not CONFIG.get("display_info_panel_on_hover", False):
                    person_to_show_details = None # Reset for current frame

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings_current_frame = face_recognition.face_encodings(rgb_frame, face_locations)

                ppe_detections_list, _ = detect_ppe_yolov8(frame)
                recognized_this_frame_details = []

                for i, current_face_encoding in enumerate(face_encodings_current_frame):
                    name = "Unknown"
                    person_id_display = "N/A"
                    metadata_for_this_person = None
                    
                    if db_known_face_encodings: # Check if there are any known faces loaded
                        matches = face_recognition.compare_faces(db_known_face_encodings, current_face_encoding, tolerance=CONFIG.get("face_recognition_tolerance", 0.6))
                        face_distances = face_recognition.face_distance(db_known_face_encodings, current_face_encoding)
                        
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                # Get metadata from the list using the best_match_index
                                metadata_for_this_person = db_known_face_metadata_list[best_match_index]
                                name = metadata_for_this_person.get('display_name', "ErrorName")
                                person_id_display = metadata_for_this_person.get("ID", "N/A") # 'ID' is the key used in draw_info_panel

                    if name == "Unknown":
                        if CONFIG.get("show_unknown_person_id", True):
                            person_id_display = f"UNKN_{unknown_id_counter}"
                        else:
                            person_id_display = "Unknown"
                        if CONFIG.get("log_unknown_detections", False):
                            print(f"Log: Unknown person detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    top, right, bottom, left = face_locations[i]
                    if name != "Unknown" and metadata_for_this_person: # Ensure metadata exists
                         detected_face_boxes_for_hover.append((name, (left, top, right, bottom)))

                    id_text_y_offset = -10
                    id_text_content = f"ID {person_id_display}"
                    (id_text_w, id_text_h), _ = cv2.getTextSize(id_text_content, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    id_bg_top_left = (left, top + id_text_y_offset - id_text_h - 5)
                    id_bg_bottom_right = (left + id_text_w + 10, top + id_text_y_offset + 5)
                    if id_bg_top_left[1] < 0:
                        id_text_y_offset = bottom + 10 
                        id_bg_top_left = (left, top + id_text_y_offset)
                        id_bg_bottom_right = (left + id_text_w + 10, top + id_text_y_offset + id_text_h + 5)

                    cv2.rectangle(frame, id_bg_top_left, id_bg_bottom_right, (0,0,0), cv2.FILLED)
                    cv2.putText(frame, id_text_content, (left + 5, top + id_text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 3)

                    recognized_this_frame_details.append({
                        "name": name, "id": person_id_display, "box": (left, top, right, bottom),
                        "metadata": metadata_for_this_person
                    })
                
                if any(p['name'] == "Unknown" for p in recognized_this_frame_details):
                    if CONFIG.get("show_unknown_person_id", True): # only increment if we were actually showing UNKN IDs
                        unknown_id_counter +=1

                for ppe_label, ppe_confidence, ppe_color, (px, py, pw, ph) in ppe_detections_list:
                    cv2.rectangle(frame, (px, py), (px + pw, py + ph), ppe_color, 2)
                    label_text = f"{ppe_label}: {ppe_confidence:.2f}"
                    text_y = py - 7 if py - 7 > 7 else py + 15
                    cv2.putText(frame, label_text, (px + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ppe_color, 1, cv2.LINE_AA)

                unknown_face_count = sum(1 for p_detail in recognized_this_frame_details if p_detail["name"] == "Unknown")
                if unknown_face_count > 0:
                    cv2.putText(frame, f"WARNING: {unknown_face_count} UNKNOWN FACE(S)", (frame_width // 2 - 200, 30),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)

                if person_to_show_details and person_to_show_details in db_display_name_to_metadata_map:
                    draw_info_panel(frame, db_display_name_to_metadata_map[person_to_show_details])
                
                draw_hud_elements(frame, frame_width)
                draw_status_bar(frame, frame_width)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()