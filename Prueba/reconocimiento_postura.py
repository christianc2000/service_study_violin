import cv2
import mediapipe as mp

def detect_body_posture(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Aplicar filtro de suavizado
    image_blurred = cv2.GaussianBlur(image_rgb, (5, 5), 0)

# Aplicar ajuste de contraste y brillo
    alpha = 1.5  # Contraste
    beta = 10    # Brillo
    image_adjusted = cv2.convertScaleAbs(image_blurred, alpha=alpha, beta=beta)

    # Inicializar el detector de pose de MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7)

    # Realizar la detección de postura corporal
    results = pose.process(image_adjusted)

    # Dibujar los landmarks en la imagen
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Guardar la imagen con los landmarks dibujados
        cv2.imwrite('annotated_image.jpg', annotated_image)

    # Liberar recursos
    pose.close()

# Ejemplo de uso
detect_body_posture('postura_parado_agarre.jpg')
# detect_body_posture('postura_parado_arco_arriba.jpg')
