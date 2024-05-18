import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Función para calcular el ángulo entre tres puntos
def calculate_angle(point1, point2, point3):
    vector1 = [point2[0] - point1[0], point2[1] - point1[1]]
    vector2 = [point2[0] - point3[0], point2[1] - point3[1]]
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_radians = math.acos(cosine_angle)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

cap = cv2.VideoCapture(0)  # Reemplazar 'nombre_del_video.mp4' con la ruta de tu video
# Variables para llevar el seguimiento de los segundos que el ángulo está en el rango deseado
angle_in_range_counter = 0
five_seconds_counter = 0

with mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(frame_rgb)
        if results.pose_landmarks is not None:
            x1 = int(results.pose_landmarks.landmark[11].x * width)
            y1 = int(results.pose_landmarks.landmark[11].y * height)
            x2 = int(results.pose_landmarks.landmark[13].x * width)
            y2 = int(results.pose_landmarks.landmark[13].y * height)
            x3 = int(results.pose_landmarks.landmark[15].x * width)
            y3 = int(results.pose_landmarks.landmark[15].y * height)
            x4 = int(results.pose_landmarks.landmark[23].x * width)
            y4 = int(results.pose_landmarks.landmark[23].y * height)
            x5 = int(results.pose_landmarks.landmark[12].x * width)
            y5 = int(results.pose_landmarks.landmark[12].y * height)
            x6 = int(results.pose_landmarks.landmark[24].x * width)
            y6 = int(results.pose_landmarks.landmark[24].y * height)
            x7 = int(results.pose_landmarks.landmark[26].x * width)
            y7 = int(results.pose_landmarks.landmark[26].y * height)
            x8 = int(results.pose_landmarks.landmark[28].x * width)
            y8 = int(results.pose_landmarks.landmark[28].y * height)
            x9 = int(results.pose_landmarks.landmark[25].x * width)
            y9 = int(results.pose_landmarks.landmark[25].y * height)
            x10 = int(results.pose_landmarks.landmark[27].x * width)
            y10 = int(results.pose_landmarks.landmark[27].y * height)
            x11 = int(results.pose_landmarks.landmark[0].x * width)
            y11 = int(results.pose_landmarks.landmark[0].y * height)
            # cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
            # cv2.line(frame, (x2, y2), (x3, y3), (255, 255, 255), 3)
            # cv2.line(frame, (x1, y1), (x4, y4), (255, 255, 255), 3)
            # cv2.line(frame, (x1, y1), (x5, y5), (255, 255, 255), 3)
            # cv2.line(frame, (x5, y5), (x6, y6), (255, 255, 255), 3)
            # cv2.line(frame, (x6, y6), (x4, y4), (255, 255, 255), 3)
            cv2.line(frame, (x6, y6), (x7, y7), (255, 255, 255), 3)
            cv2.line(frame, (x7, y7), (x8, y8), (255, 255, 255), 3)
            cv2.line(frame, (x4, y4), (x9, y9), (255, 255, 255), 3)
            cv2.line(frame, (x9, y9), (x10, y10), (255, 255, 255), 3)
            cv2.line(frame, (x1, y1), (x11, y11), (255, 255, 255), 3)
            cv2.line(frame, (x5, y5), (x11, y11), (255, 255, 255), 3)
            
            angle = calculate_angle((x1, y1), (x2, y2), (x3, y3))
            angle2 = calculate_angle((x5, y5), (x1, y1), (x2, y2))
            angle3 = calculate_angle((x4, y4), (x9, y9), (x10, y10))
            angle4 = calculate_angle((x11, y11), (x1, y1), (x4, y4))
            # cv2.putText(frame, f"Angle brazo izquierdo: {angle:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, f"Angle hombre codo: {angle2:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
             # Cambiar color de las líneas según los ángulos
            if 75 <= angle <= 90 and 120 <= angle2 <= 130:
                line_color = (0, 255, 0)  # Verde
                angle_in_range_counter += 1
                if angle_in_range_counter >= 5 * 30:  # 30 fps, por lo que 5 segundos son 5 * 30 frames
                    five_seconds_counter += 1
                    angle_in_range_counter = 0
            else:
                line_color = (0, 0, 255)  # Rojo
                angle_in_range_counter = 0
            
            # Dibujar líneas con el color correspondiente
            cv2.line(frame, (x1, y1), (x2, y2), line_color, 3)
            cv2.line(frame, (x2, y2), (x3, y3), line_color, 3)
            cv2.line(frame, (x1, y1), (x4, y4), line_color, 3)
            cv2.line(frame, (x1, y1), (x5, y5), line_color, 3)
            cv2.line(frame, (x5, y5), (x6, y6), line_color, 3)
            cv2.line(frame, (x6, y6), (x4, y4), line_color, 3)
            
            # Mostrar ángulos en pantalla
            cv2.putText(frame, f"Angle brazo izquierdo: {angle:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle hombre codo: {angle2:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle pierna izquierda: {angle3:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle nariz cadera izq: {angle4:.2f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Segundos en rango: {angle_in_range_counter / 30}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Veces que aguantó 5 segundos: {five_seconds_counter}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
