import cv2
import os
import numpy as np

# Criar reconhecedor LBPH
reconhecedor = cv2.face.LBPHFaceRecognizer_create()

# Treinar com imagens da pasta 'rostos'
# (cada pessoa com ID diferente)
imagens = []
ids = []

for pessoa_id, nome_pasta in enumerate(os.listdir("rostos")):
    for arquivo in os.listdir(f"rostos/{nome_pasta}"):
        img = cv2.imread(f"rostos/{nome_pasta}/{arquivo}", cv2.IMREAD_GRAYSCALE)
        imagens.append(img)
        ids.append(pessoa_id)

reconhecedor.train(imagens, np.array(ids))

# Captura da webcam e reconhecimento
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        rosto = gray[y:y+h, x:x+w]
        id_, conf = reconhecedor.predict(rosto)
        if conf < 50:  # quanto menor, melhor o reconhecimento
            nome = os.listdir("rostos")[id_]
        else:
            nome = "Desconhecido"
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, nome, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Reconhecimento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
