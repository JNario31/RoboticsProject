import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from matrix_utils import compute_transformation, apply_transformation

# --- CONSTANTS --- 
# blue square has constant distance from base of kinova arm
BLUE_SQUARE_XW = 0.075
BLUE_SQUARE_YW = 0.265
WORLD_POINTS = [[BLUE_SQUARE_XW, BLUE_SQUARE_YW]]

# -------- Inference --------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dOXf27URLjdeZMgyJ7en"
)

result = CLIENT.infer("test_image_2.jpg", model_id="cube-color-gzmh4/14")

# -------- Load image --------
img = cv2.imread("test_image_2.jpg")

# -------- Draw predictions --------
preds = result['predictions']

for pred in preds:
    x = int(pred['x'])
    y = int(pred['y'])
    w = int(pred['width'])
    h = int(pred['height'])
    class_name = pred['class']
    conf = pred['confidence']
    if(class_name == "blue"):
        blue_square_xp = x
        blue_square_yp = y
        pixel_points = [[blue_square_xp, blue_square_yp]]

    # Convert from center-x/y to top-left corner
    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label
    label = f"{class_name} ({conf:.2f})"
    cv2.putText(img, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
# --- Compute transformation matrix ---
print("Using blue square to compute transformation matrix...")
matrix = compute_transformation(pixel_points, WORLD_POINTS)
print("Matrix computed:")
print(matrix)

# --- Use transformation matrix to find real world coordinates of cubes ---
for pred in preds:
    xp = int(pred['x'])
    yp = int(pred['y'])
    class_name = pred['class']
    conf = pred['confidence']
    point = xp,yp
    xw, yw = apply_transformation(matrix, point)
    print("Real World Coordinates For " + class_name + " " + str(conf))
    print("x = " + str(xw))
    print("y = " + str(yw))


# -------- Show or save --------
#cv2.imshow("Predictions", img)
#cv2.waitKey(0)

# Optional: save output
cv2.imwrite("output.jpg", img)
