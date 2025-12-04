from fastapi import FastAPI, UploadFile
import uvicorn
import cv2
import numpy as np
import joblib
import mediapipe as mp


# ---------------------------------------------------
# 1. Cargar modelo
# ---------------------------------------------------
model = joblib.load("sign_language_model.pkl")

labels_dict = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

# ---------------------------------------------------
# 2. Inicializar MediaPipe
# ---------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------------
# 3. Backend FastAPI
# ---------------------------------------------------
app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile):

    # Leer la imagen enviada desde el navegador
    img_bytes = await file.read()

    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return {"letter": "None"}

    # Extraer landmarks
    hand_landmarks = results.multi_hand_landmarks[0]

    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    X = np.array(landmarks).reshape(1, -1)

    pred = model.predict(X)[0]
    letter = labels_dict[int(pred)]

    return {"letter": letter}


# ---------------------------------------------------
# 4. Iniciar servidor
# ---------------------------------------------------
if __name__ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=8000)
