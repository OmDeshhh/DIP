import cv2
import numpy as np
import os
from tkinter import Tk, Label, Button, filedialog, Canvas, StringVar
from PIL import Image, ImageTk

# Output directory
OUTPUT_DIR = "processed_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_processed_image = None

def estimate_temperature(average_value):
    # Estimated temperature based on the pixel intensity
    return round(36 + ((average_value - 200) / 55) * 4, 1)

def draw_temp_scale(image):
    # Draw a color scale for temperature visualization
    scale = np.zeros((240, 30, 3), dtype=np.uint8)
    for i in range(240):
        h = int(10 * (240 - i) / 240)
        color = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        scale[i] = color

    cv2.putText(scale, "40°C", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(scale, "36°C", (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    image[:, -30:] = scale
    return image

def process_thermal_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))

    # Detect face and draw square box
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        side = max(w, h)  # Make the box square
        cv2.rectangle(image, (x, y), (x + side, y + side), (255, 255, 0), 2)

    upper_body = image[:240, :]
    smoothed = cv2.GaussianBlur(upper_body, (5, 5), 0)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)

    # Adjust HSV color ranges for better temperature detection
    lower_hot = np.array([0, 80, 160])
    upper_hot = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_hot, upper_hot)

    # Clean up the mask using morphological transformations
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Sharpen the image using Laplacian filter
    laplacian = cv2.Laplacian(upper_body, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(laplacian)
    combined = cv2.addWeighted(upper_body, 0.8, sharpened, 0.2, 0)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = combined.copy()

    fever_detected = False
    estimated_temp = 36.0

    # Process contours and calculate temperature
    for contour in contours:
        if 800 < cv2.contourArea(contour) < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            roi = hsv[y:y + h, x:x + w, 2]
            avg_value = np.mean(roi)
            estimated_temp = estimate_temperature(avg_value)

            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(output, f"{estimated_temp}°C", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if estimated_temp > 38.0:
                fever_detected = True

    output = draw_temp_scale(output)
    return output, estimated_temp, fever_detected

def upload_image():
    global last_processed_image
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if file_path:
        result, temp, fever = process_thermal_image(file_path)
        last_processed_image = result

        # Update image on canvas
        img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((640, 240))
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        canvas.image = img_tk

        # Update status label
        status = "FEVER DETECTED" if fever else "NORMAL"
        color = "red" if fever else "green"
        status_text_var.set(f"{status} ({temp}°C)")
        status_label.config(fg=color)

def download_image():
    global last_processed_image
    if last_processed_image is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG Image", "*.jpg")])
        if save_path:
            cv2.imwrite(save_path, last_processed_image)

# GUI Setup
root = Tk()
root.title("Thermal Fever Detection")
root.geometry("660x400")

status_text_var = StringVar()
status_text_var.set("No image uploaded")

Label(root, text="Upload and Download Processed Thermal Image", font=("Helvetica", 14)).pack(pady=10)

canvas = Canvas(root, width=640, height=240)
canvas.pack()

status_label = Label(root, textvariable=status_text_var, font=("Helvetica", 12, "bold"))
status_label.pack(pady=5)

Button(root, text="Upload Image", command=upload_image, width=20).pack(pady=3)
Button(root, text="Download Result", command=download_image, width=20).pack()

root.mainloop()
