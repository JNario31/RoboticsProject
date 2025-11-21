import cv2
from inference_sdk import InferenceHTTPClient

# -------- Inference --------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dOXf27URLjdeZMgyJ7en"
)

result = CLIENT.infer("test_image_4.jpg", model_id="cube-color-gzmh4/14")

# -------- Load image --------
img = cv2.imread("test_image_4.jpg")

# -------- Draw predictions --------
preds = result['predictions']

for pred in preds:
    x = int(pred['x'])
    y = int(pred['y'])
    w = int(pred['width'])
    h = int(pred['height'])
    class_name = pred['class']
    conf = pred['confidence']
    print(f'x: {x}')
    print(f'y: {y}')
    print(f'Image width: {w}')
    print(f'Image height: {h}')
    print(f'Class name: {class_name}')
    print(f'Confidence: {conf}')

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

# -------- Show or save --------
cv2.imshow("Predictions", img)
cv2.waitKey(0)

# Optional: save output
cv2.imwrite("output.jpg", img)
