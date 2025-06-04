import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import face_recognition
import datetime
import shutil
import sqlite3
import pickle
from PIL import Image, ImageTk

class RegistrationForm:
    def __init__(self, root):
        self.root = root
        self.root.title("Person Registration Form")
        self.root.geometry("600x700")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Variables
        self.display_name_var = tk.StringVar()
        self.age_var = tk.StringVar()
        self.function_var = tk.StringVar()
        self.person_id_var = tk.StringVar()
        self.hashcode_var = tk.StringVar()
        self.image_path = tk.StringVar()
        
        # Form fields
        ttk.Label(main_frame, text="Person Registration", font=('Helvetica', 16, 'bold')).grid(row=0, column=0, columnspan=2, pady=20)
        
        # Display Name
        ttk.Label(main_frame, text="Display Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.display_name_var, width=40).grid(row=1, column=1, pady=5)
        
        # Age
        ttk.Label(main_frame, text="Age:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.age_var, width=40).grid(row=2, column=1, pady=5)
        
        # Function/Role
        ttk.Label(main_frame, text="Function/Role:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.function_var, width=40).grid(row=3, column=1, pady=5)
        
        # Person ID
        ttk.Label(main_frame, text="Person ID Code:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.person_id_var, width=40).grid(row=4, column=1, pady=5)
        
        # Hashcode
        ttk.Label(main_frame, text="Hashcode (optional):").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.hashcode_var, width=40).grid(row=5, column=1, pady=5)
        
        # Image Selection
        ttk.Label(main_frame, text="Profile Image:").grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Button(main_frame, text="Browse Image", command=self.browse_image).grid(row=6, column=1, sticky=tk.W, pady=5)
        
        # Image preview
        self.preview_label = ttk.Label(main_frame)
        self.preview_label.grid(row=7, column=0, columnspan=2, pady=20)
        
        # Submit Button
        ttk.Button(main_frame, text="Register Person", command=self.register_person).grid(row=8, column=0, columnspan=2, pady=20)
        
        # Status Label
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=9, column=0, columnspan=2, pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.image_path.set(file_path)
            # Show image preview
            self.show_image_preview(file_path)

    def show_image_preview(self, image_path):
        try:
            # Open and resize image for preview
            image = Image.open(image_path)
            image.thumbnail((200, 200))  # Resize for preview
            photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image preview: {str(e)}")

    def register_person(self):
        # Validate inputs
        if not all([self.display_name_var.get(), self.person_id_var.get(), self.image_path.get()]):
            messagebox.showerror("Error", "Display Name, Person ID, and Image are required!")
            return
            
        try:
            age = int(self.age_var.get()) if self.age_var.get() else None
        except ValueError:
            messagebox.showerror("Error", "Age must be a number!")
            return
            
        try:
            # Load and process image
            image = face_recognition.load_image_file(self.image_path.get())
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                messagebox.showerror("Error", "No face found in the image!")
                return
                
            face_encoding = encodings[0]
            
            # Create unique filename
            base_filename = os.path.basename(self.image_path.get())
            _, ext_part = os.path.splitext(base_filename)
            sanitized_name = "".join(c if c.isalnum() else "_" for c in self.display_name_var.get())
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            new_image_filename = f"{sanitized_name}_{self.person_id_var.get()}_{timestamp}{ext_part}"
            
            # Ensure registered_images directory exists
            if not os.path.exists("registered_images"):
                os.makedirs("registered_images")
                
            # Copy image file
            destination_path = os.path.join("registered_images", new_image_filename)
            shutil.copy(self.image_path.get(), destination_path)
            
            # Save to database
            self.add_person_to_db(
                self.display_name_var.get(),
                age,
                self.function_var.get(),
                self.person_id_var.get(),
                self.hashcode_var.get(),
                new_image_filename,
                face_encoding
            )
            
            messagebox.showinfo("Success", "Person registered successfully!")
            self.clear_form()
            
        except Exception as e:
            messagebox.showerror("Error", f"Registration failed: {str(e)}")

    def add_person_to_db(self, display_name, age, function_text, person_id_code, hashcode, image_filename, face_encoding):
        conn = sqlite3.connect("personnel_data.db")
        cursor = conn.cursor()
        
        # Create table if not exists
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
        
        try:
            serialized_encoding = pickle.dumps(face_encoding)
            cursor.execute("""
            INSERT INTO persons (display_name, age, function_text, person_id_code, hashcode, image_filename, face_encoding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (display_name, age, function_text, person_id_code, hashcode, image_filename, serialized_encoding))
            conn.commit()
        except sqlite3.IntegrityError:
            raise Exception(f"Person ID Code '{person_id_code}' already exists in the database.")
        finally:
            conn.close()

    def clear_form(self):
        self.display_name_var.set("")
        self.age_var.set("")
        self.function_var.set("")
        self.person_id_var.set("")
        self.hashcode_var.set("")
        self.image_path.set("")
        self.preview_label.configure(image="")
        self.status_label.configure(text="")

def main():
    root = tk.Tk()
    app = RegistrationForm(root)
    root.mainloop()

if __name__ == "__main__":
    main()