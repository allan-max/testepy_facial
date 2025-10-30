import cv2
import os
import numpy as np

BASE_DIR = "rostos"
TAMANHO_ROSTO = (200, 200)
NUM_FOTOS = 40
LIMIAR_CONF = 80

def cadastrar_rosto():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    nome = input("Digite o nome da pessoa: ").strip()
    pasta_pessoa = os.path.join(BASE_DIR, nome)
    if not os.path.exists(pasta_pessoa):
        os.makedirs(pasta_pessoa)
    else:
        print("⚠️ Essa pessoa já existe, as novas fotos serão adicionadas.")
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    contador = 0
    print("\n➡️ Mova a cabeça lentamente: olhe para os lados, para cima e para baixo.")
    print("➡️ O sistema vai capturar automaticamente as imagens.\n")
    print("Pressione 'q' para parar.\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro ao acessar a webcam.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            rosto = gray[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, TAMANHO_ROSTO)
            contador += 1
            caminho_foto = os.path.join(pasta_pessoa, f"{contador}.jpg")
            cv2.imwrite(caminho_foto, rosto)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturando {contador}/{NUM_FOTOS}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Captura de Rosto", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❌ Captura interrompida manualmente.")
            break
        if contador >= NUM_FOTOS:
            print(f"\n✅ Captura concluída: {contador} imagens salvas em '{pasta_pessoa}'.")
            break
    cap.release()
    cv2.destroyAllWindows()

def reconhecer_rostos():
    if not os.path.exists(BASE_DIR):
        print("❌ Nenhuma pasta de rostos encontrada. Cadastre alguém primeiro.")
        return
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    imagens, ids = [], []
    for pessoa_id, nome_pasta in enumerate(os.listdir(BASE_DIR)):
        caminho_pasta = os.path.join(BASE_DIR, nome_pasta)
        if not os.path.isdir(caminho_pasta):
            continue
        for arquivo in os.listdir(caminho_pasta):
            img_path = os.path.join(caminho_pasta, arquivo)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, TAMANHO_ROSTO)
            imagens.append(img)
            ids.append(pessoa_id)
    if len(imagens) < 2:
        print("❌ Poucas imagens encontradas. Cadastre mais rostos antes de treinar.")
        return
    reconhecedor.train(imagens, np.array(ids))
    nomes = os.listdir(BASE_DIR)
    print("✅ Treinamento concluído. Iniciando reconhecimento...\n")
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            rosto = gray[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, TAMANHO_ROSTO)
            id_, conf = reconhecedor.predict(rosto)
            nome = nomes[id_] if conf < LIMIAR_CONF else "Desconhecido"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{nome} ({int(conf)})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def menu():
    while True:
        print("\n=== SISTEMA DE RECONHECIMENTO FACIAL ===")
        print("[1] Cadastrar novo rosto")
        print("[2] Iniciar reconhecimento")
        print("[3] Sair")
        opcao = input("Escolha uma opção: ")
        if opcao == "1":
            cadastrar_rosto()
        elif opcao == "2":
            reconhecer_rostos()
        elif opcao == "3":
            print("👋 Encerrando o programa.")
            break
        else:
            print("❌ Opção inválida, tente novamente.")

if __name__ == "__main__":
    menu()