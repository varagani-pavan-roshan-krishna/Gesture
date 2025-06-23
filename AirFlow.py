import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import math
from tensorflow.keras.models import load_model

# Load your trained model and class names
model = load_model("quickdraw_model.h5")
class_names = [
    'apple', 'banana', 'bed', 'bird', 'book', 'bottlecap', 'bread', 'bus', 'cake', 'car',
    'cat', 'chair', 'cloud', 'cookie', 'cup', 'dog', 'door', 'duck', 'ear', 'envelope',
    'eye', 'face', 'fan', 'fish', 'flower', 'fork', 'hand', 'hat', 'ice cream', 'key',
    'knife', 'ladder', 'leaf', 'light bulb', 'pencil', 'pizza',
    'rabbit', 'rainbow', 'shoe', 'smiley face', 'sock', 'spoon', 'star', 'sun', 'toothbrush',
    'tree', 'umbrella', 'mountain'
]

canvas_height, canvas_width = 471, 636
paintWindow = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8) + 30

colors = [
    (80, 0, 0),      # dark blue
    (0, 80, 0),      # dark green
    (0, 0, 80),      # dark red
    (0, 80, 80)      # dark yellow/cyan
]
colorIndex = 0

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
blue_index = green_index = red_index = yellow_index = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prediction_label = ""

def preprocess_for_model(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Use the version with more white pixels (more drawing)
    if np.sum(thresh_inv == 255) > np.sum(thresh == 255):
        thresh_used = thresh_inv
    else:
        thresh_used = thresh
    coords = cv2.findNonZero(thresh_used)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        size = max(w, h)
        x_pad = (size - w) // 2
        y_pad = (size - h) // 2
        cropped = thresh_used[max(y - y_pad, 0):min(y + h + y_pad, thresh_used.shape[0]),
                              max(x - x_pad, 0):min(x + w + x_pad, thresh_used.shape[1])]
        resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        resized = np.zeros((28, 28), dtype=np.uint8)
    norm = resized / 255.0
    return norm.reshape(1, 28, 28, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (canvas_width, canvas_height))
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw UI on frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (80, 0, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 80, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 80), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 80, 80), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * canvas_width)
                lmy = int(lm.y * canvas_height)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        if len(landmarks) >= 9:
            index_tip = (landmarks[8][0], landmarks[8][1])
            thumb_tip = (landmarks[4][0], landmarks[4][1])
            center = index_tip
            cv2.circle(frame, center, 8, (0, 255, 255), -1)
            dist = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

            if dist < 40:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
            elif center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear Button
                    bpoints = [deque(maxlen=1024)]
                    gpoints = [deque(maxlen=1024)]
                    rpoints = [deque(maxlen=1024)]
                    ypoints = [deque(maxlen=1024)]
                    blue_index = green_index = red_index = yellow_index = 0
                    paintWindow[67:, :, :] = 30
                elif 160 <= center[0] <= 255:
                    colorIndex = 0  # Blue
                elif 275 <= center[0] <= 370:
                    colorIndex = 1  # Green
                elif 390 <= center[0] <= 485:
                    colorIndex = 2  # Red
                elif 505 <= center[0] <= 600:
                    colorIndex = 3  # Yellow
            else:
                bpoints.append(deque(maxlen=1024))
                blue_index += 1
                gpoints.append(deque(maxlen=1024))
                green_index += 1
                rpoints.append(deque(maxlen=1024))
                red_index += 1
                ypoints.append(deque(maxlen=1024))
                yellow_index += 1
    else:
        bpoints.append(deque(maxlen=1024))
        blue_index += 1
        gpoints.append(deque(maxlen=1024))
        green_index += 1
        rpoints.append(deque(maxlen=1024))
        red_index += 1
        ypoints.append(deque(maxlen=1024))
        yellow_index += 1

    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 4)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 4)

    # --- Recognition on 'r' key ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        # Crop out the header (first 70 rows)
        sketch_img = paintWindow[70:, :, :]
        sketch_input = preprocess_for_model(sketch_img)
        # Optional: visualize input to model
        cv2.imshow("Preprocessed", (sketch_input[0]*255).astype(np.uint8))
        cv2.waitKey(500)  # Show for 0.5 seconds
        cv2.destroyWindow("Preprocessed")
        pred = model.predict(sketch_input)
        pred_label = np.argmax(pred)
        prediction_label = class_names[pred_label]
        print("You drew:", prediction_label)

    # Show prediction label if available
    if prediction_label:
        cv2.putText(frame, f"Prediction: {prediction_label}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
