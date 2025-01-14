import cv2
from ultralytics import YOLO
import simpleaudio as sa
import os
import numpy as np

model = YOLO("fire_and_smoke.pt")

def create_window(window_name="Fire and Smoke Detection System", width=1280, height=720):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    return window_name

def add_status_bar(image, status_text="No Fire Detected", is_alert=False):
    h, w = image.shape[:2]
    status_bar = np.zeros((60, w, 3), dtype=np.uint8)
    if is_alert:
        status_bar[:] = (0, 0, 255)  
        text_color = (255, 255, 255)  
    else:
        status_bar[:] = (0, 255, 0)  
        text_color = (0, 0, 0) 
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(status_text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (60 + text_size[1]) // 2
    
    cv2.putText(status_bar, status_text, (text_x, text_y), font, font_scale, text_color, thickness)
    return np.vstack((image, status_bar))

def add_info_overlay(image, info_text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    padding = 10
    
    for i, line in enumerate(info_text.split('\n')):
        y = 30 + i * 25
        cv2.putText(image, line, (padding, y), font, font_scale, (255, 255, 255), thickness + 1)
        cv2.putText(image, line, (padding, y), font, font_scale, (0, 0, 0), thickness)
    
    return image

def play_alarm():
    try:
        wave_obj = sa.WaveObject.from_wave_file("test.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing alarm: {e}")

def fix_path(path):
    path = path.strip()
    path = path.replace('\\', '')
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path

def check_permission(path):
    try:
        fixed_path = fix_path(path)
        print(f"Trying to access: {fixed_path}")
        
        if not os.path.exists(fixed_path):
            print(f"File not found: {fixed_path}")
            print("Please check if the path is correct and the file exists")
            return False, None
            
        if not os.access(fixed_path, os.R_OK):
            print(f"Cannot access file: {fixed_path}")
            print("Try running: chmod +r '" + fixed_path + "'")
            return False, None
            
        return True, fixed_path
    except Exception as e:
        print(f"Error checking file: {e}")
        return False, None

def process_image(image_path):
    permission_result = check_permission(image_path)
    if not permission_result[0]:
        return
    
    fixed_path = permission_result[1]
    window_name = create_window()
    
    try:
        print(f"Loading image from: {fixed_path}")
        img = cv2.imread(fixed_path)
        if img is None:
            print(f"Error: Unable to load image from {fixed_path}")
            return
            
        res = model(img)
        fire_detected = len(res[0].boxes) > 0
        
        annotated_img = res[0].plot()
        info_text = f"File: {os.path.basename(fixed_path)}\nPress 'ESC' to exit"
        annotated_img = add_info_overlay(annotated_img, info_text)
        
        if fire_detected:
            display_img = add_status_bar(annotated_img, "FIRE/SMOKE DETECTED!", True)
            play_alarm()
        else:
            display_img = add_status_bar(annotated_img, "No Fire Detected", False)
            
        cv2.imshow(window_name, display_img)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  
                break
                
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error processing image: {e}")

def process_video(video_path):
    permission_result = check_permission(video_path)
    if not permission_result[0]:
        return
        
    fixed_path = permission_result[1]
    window_name = create_window()
    
    try:
        cap = cv2.VideoCapture(fixed_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video from {fixed_path}")
            return
            
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            res = model(frame)
            fire_detected = len(res[0].boxes) > 0
            
            annotated_frame = res[0].plot()
            info_text = f"File: {os.path.basename(fixed_path)}\nFrame: {frame_count}\nPress 'ESC' to exit"
            annotated_frame = add_info_overlay(annotated_frame, info_text)
            
            if fire_detected:
                display_frame = add_status_bar(annotated_frame, "FIRE/SMOKE DETECTED!", True)
                play_alarm()
            else:
                display_frame = add_status_bar(annotated_frame, "No Fire Detected", False)
                
            cv2.imshow(window_name, display_frame)
            
            if cv2.waitKey(1) == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error processing video: {e}")

def process_webcam():
    window_name = create_window()
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to access webcam")
            return
            
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            res = model(frame)
            fire_detected = len(res[0].boxes) > 0
            
            annotated_frame = res[0].plot()
            info_text = f"Webcam Feed\nFrame: {frame_count}\nPress 'ESC' to exit"
            annotated_frame = add_info_overlay(annotated_frame, info_text)
            
            if fire_detected:
                display_frame = add_status_bar(annotated_frame, "FIRE/SMOKE DETECTED!", True)
                play_alarm()
            else:
                display_frame = add_status_bar(annotated_frame, "No Fire Detected", False)
                
            cv2.imshow(window_name, display_frame)
            
            if cv2.waitKey(1) == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error accessing webcam: {e}")

def main():
    while True:
        print("\nFire and Smoke Detection System")
        print("-" * 30)
        print("1. Process image")
        print("2. Process video")
        print("3. Use webcam")
        print("4. Quit")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            path = input("Enter image path: ")
            process_image(path)
        elif choice == '2':
            path = input("Enter video path: ")
            process_video(path)
        elif choice == '3':
            process_webcam()
        elif choice == '4':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please enter a number between 1-4.")

if __name__ == "__main__":
    main()