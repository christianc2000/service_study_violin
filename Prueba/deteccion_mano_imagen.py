import cv2
import mediapipe as mp

# Inicializar el módulo HandLandmarker de MediaPipe
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Cargar la imagen
image_path = "mano_agarre_1.jpg"  # Reemplaza "tu_imagen.jpg" con la ruta de tu imagen
image = cv2.imread(image_path)

# Convertir la imagen de BGR a RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detectar manos en la imagen
results = mp_hands.process(image_rgb)

# Dibujar los landmarks de las manos si se detectan manos
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Dibujar los landmarks de la mano
        mp.solutions.drawing_utils.draw_landmarks(
            image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

# Mostrar la imagen con los landmarks de las manos
cv2.imshow('Hand Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
