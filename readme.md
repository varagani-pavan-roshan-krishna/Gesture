# ✍️ Hand Gesture Drawing & Recognition System

This is an interactive computer vision project that lets you draw on a virtual canvas using just your **hand gestures**, and then recognizes what you've drawn using a pre-trained deep learning model (trained on Google's [QuickDraw dataset](https://quickdraw.withgoogle.com/data)). No mouse, no pen — just your **fingertips** and a webcam.

---

## 🔧 What It Does

* Tracks your **hand movements** using **MediaPipe**.
* Lets you draw with your index finger, choose colors, and even clear the canvas.
* Uses a trained model to **recognize** your sketch (over 50 object classes like *cat, sun, pizza, tree,* etc.).
* Just press **'r'** to get the prediction and **'q'** to quit.

## DEMO
https://github.com/user-attachments/assets/4cc7ea57-1b92-41b9-a58e-48be58a76421

## 🧠 Technologies Used

* **OpenCV** – For webcam access and UI.
* **MediaPipe** – For real-time hand tracking.
* **TensorFlow / Keras** – To load and run the drawing recognition model.
* **NumPy** – For image preprocessing and manipulation.

---

## 🚀 How to Run

1. Install dependencies:

   ```bash
   pip install opencv-python mediapipe numpy tensorflow
   ```

2. Place your trained model as `quickdraw_model.h5` in the working directory.

3. Run the script:

   ```bash
   python hand_draw_recognizer.py
   ```

4. Use your **index finger** to draw. Use your **thumb** to click buttons on the top of the screen.

---

## 🎨 Features

* ✋ Draw using gestures
* 🌈 Color palette (Blue, Green, Red, Yellow)
* 🧽 Clear canvas button
* 🧠 Real-time object recognition with **confidence scores**
* 📎 Built using the [QuickDraw Dataset by Google](https://quickdraw.withgoogle.com/data)

---

## 📌 Controls

| Gesture or Key             | Action                |
| -------------------------- | --------------------- |
| Point finger & pinch thumb | Start drawing         |
| Move hand to top buttons   | Select color or clear |
| Press **'r'**              | Run recognition       |
| Press **'q'**              | Quit app              |

---

## 🖼 Sample Output

```
You drew: cat
Prediction: 🐱 Cat (Confidence: 94.3%)
```

---

## 🙌 Credits

Built using Google's **MediaPipe** for hand tracking and a **CNN** trained on the [QuickDraw dataset](https://quickdraw.withgoogle.com/data).

---
