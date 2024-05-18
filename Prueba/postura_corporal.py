import cv2
import mediapipe as mp

# Función para dibujar landmarks de pose en una imagen
def draw_landmarks_on_image(image, results):
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image

# Inicializar los módulos de MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Crear un objeto PoseLandmarker
pose_landmarker = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Capturar el video desde un archivo de video o desde una cámara web
cap = cv2.VideoCapture(0)  # Reemplaza 'nombre_del_video.mp4' con el nombre de tu archivo de video o usa 0 para capturar desde la cámara web

while cap.isOpened():
    # Leer un fotograma del video
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el fotograma de BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar landmarks de pose en el fotograma
    results = pose_landmarker.process(frame_rgb)

    # Dibujar los landmarks de pose en el fotograma
    frame = draw_landmarks_on_image(frame, results)

    # Mostrar el fotograma con los landmarks de pose
    cv2.imshow('Pose Landmarks', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
