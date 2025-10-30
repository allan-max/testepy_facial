import cv2
import os

# === CONFIGURAÇÕES ===
base_dir = "rostos"       # pasta onde os rostos serão salvos
num_fotos = 30            # número de fotos a capturar por pessoa
tamanho = (200, 200)      # tamanho padrão para treino

# === CRIA A PASTA PRINCIPAL SE NÃO EXISTIR ===
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# === PEDE O NOME DA PESSOA ===
nome = input("Digite o nome da pessoa: ").strip()
pasta_pessoa = os.path.join(base_dir, nome)

if not os.path.exists(pasta_pessoa):
    os.makedirs(pasta_pessoa)
else:
    print("⚠️ Essa pessoa já tem uma pasta, as novas imagens serão adicionadas lá.")

# === INICIA A CAPTURA ===
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

contador = 0
print("\n➡️ Posicione o rosto em frente à câmera. Pressione 'q' para sair.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Erro ao acessar a webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        rosto = gray[y:y+h, x:x+w]
        rosto = cv2.resize(rosto, tamanho)

        contador += 1
        caminho_foto = os.path.join(pasta_pessoa, f"{contador}.jpg")
        cv2.imwrite(caminho_foto, rosto)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Capturando {contador}/{num_fotos}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Captura de Rostos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❌ Captura interrompida manualmente.")
        break

    if contador >= num_fotos:
        print(f"\n✅ Captura concluída! {num_fotos} imagens salvas em {pasta_pessoa}\n")
        break

cap.release()
cv2.destroyAllWindows()
