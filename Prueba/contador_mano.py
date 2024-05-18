import cv2
import mediapipe as mp

# Definir constantes
THUMB_INDEX_VERTICAL_THRESHOLD = 0.1  # Umbral vertical para la posición del pulgar respecto al índice
CLOSED_STATE_DURATION = 10  # Duración mínima (en fotogramas) para considerar la mano cerrada como válida

# Inicializar el módulo HandLandmarker de MediaPipe
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Capturar el video desde una cámara web
cap = cv2.VideoCapture(0)

# Inicializar variables
hand_closed_frames = 0
hand_state = "Unknown"
hand_open_count = 0  # Contador de veces que se ha abierto la mano

while cap.isOpened():
    # Leer un fotograma del video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir el fotograma de BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detectar manos en el fotograma
    results = mp_hands.process(frame_rgb)
    
    # Verificar si se detectaron manos y obtener las coordenadas de los landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]  # Punta del pulgar
            index_tip = hand_landmarks.landmark[8]  # Punta del dedo índice
            
            # Comparar las posiciones verticales de la punta del pulgar y el índice
            thumb_index_vertical_diff = thumb_tip.y - index_tip.y
            
            # Actualizar el estado de la mano
            if thumb_index_vertical_diff > THUMB_INDEX_VERTICAL_THRESHOLD:
                if hand_state != "Open":
                    hand_closed_frames = 0
                    hand_state = "Open"
                    print("Hand opened")
            else:
                hand_closed_frames += 1
                if hand_closed_frames >= CLOSED_STATE_DURATION and hand_state == "Open":
                    hand_open_count += 1
                    hand_state = "Closed"
                    print("Hand closed")
            
            # Dibujar los landmarks de la mano
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())

    # Mostrar el número de veces que se abrió la mano en la ventana
    cv2.putText(frame, f"Hand open count: {hand_open_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostrar el fotograma con los landmarks de las manos
    cv2.imshow('Hand Landmarks', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
