#!/usr/bin/env python3
"""
Camera Calibration Image Capture Script
Captures images of a chessboard pattern for camera calibration using OpenCV
"""

import cv2
import os
from datetime import datetime

# Configuration
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners (columns, rows)
IMAGE_SAVE_DIR = "calibration_images"
MIN_IMAGES = 20  # Recommended minimum number of calibration images
WINDOW_NAME = "Camera Calibration - Press SPACE to capture, Q to quit"

def main():
    # Create directory for saving images
    if not os.path.exists(IMAGE_SAVE_DIR):
        os.makedirs(IMAGE_SAVE_DIR)
        print(f"Created directory: {IMAGE_SAVE_DIR}")
    
    # Initialize camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    image_count = 0
    
    print("\n" + "="*60)
    print("Camera Calibration Image Capture")
    print("="*60)
    print(f"Chessboard size: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
    print(f"Target images: {MIN_IMAGES}")
    print("\nInstructions:")
    print("  - Move the chessboard to different positions and angles")
    print("  - Press SPACE to capture an image when corners are detected")
    print("  - Press Q to quit")
    print("  - Green corners = detected, ready to capture")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Convert to grayscale for corner detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret_corners, corners = cv2.findChessboardCorners(
            gray, 
            CHESSBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # Draw the corners if found
        display_frame = frame.copy()
        if ret_corners:
            cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_corners)
            
            # Add text indicating corners are detected
            cv2.putText(
                display_frame, 
                "Corners Detected - Press SPACE to capture", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
        else:
            cv2.putText(
                display_frame, 
                "No corners detected - Adjust chessboard position", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
        
        # Display image count
        cv2.putText(
            display_frame,
            f"Images captured: {image_count}/{MIN_IMAGES}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Show the frame
        cv2.imshow(WINDOW_NAME, display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar
            if ret_corners:
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"calibration_{timestamp}_{image_count:03d}.jpg"
                filepath = os.path.join(IMAGE_SAVE_DIR, filename)
                
                cv2.imwrite(filepath, frame)
                image_count += 1
                
                print(f"Captured image {image_count}: {filename}")
                
                if image_count >= MIN_IMAGES:
                    print(f"\n✓ Captured {MIN_IMAGES} images. You can press Q to quit or continue capturing more.")
            else:
                print("Warning: No corners detected. Cannot save image.")
        
        elif key == ord('q') or key == ord('Q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print(f"Total images captured: {image_count}")
    print(f"Images saved in: {IMAGE_SAVE_DIR}/")
    
    if image_count >= MIN_IMAGES:
        print("\n✓ You have enough images for calibration!")
        print("Next step: Run the calibration script with these images.")
    else:
        print(f"\n⚠ Warning: Only {image_count} images captured.")
        print(f"Recommended: At least {MIN_IMAGES} images for good calibration.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()