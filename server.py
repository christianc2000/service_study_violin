from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import math
import numpy as np
import os

app = Flask(__name__)

# Inicializar el módulo PoseLandmarker de MediaPipe
mp_pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.75, min_tracking_confidence=0.75)

def get_pose_landmarks_coordinates(results):
    landmarks = results.pose_landmarks.landmark
    landmarks_coordinates = [(landmark.x, landmark.y) for landmark in landmarks]
    return landmarks_coordinates

def calculate_angle(point1, point2, point3):
    vector1 = [point2[0] - point1[0], point2[1] - point1[1]]
    vector2 = [point2[0] - point3[0], point2[1] - point3[1]]
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    
    if magnitude1 * magnitude2 == 0:
        return None
    
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    
    if cosine_angle < -1 or cosine_angle > 1:
        return None
    
    angle_radians = math.acos(cosine_angle)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def analyze_practica1(image_file, form):
    # Leer la imagen de la solicitud
    image_np = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Convertir la imagen de BGR a RGB
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detectar posturas corporales en la imagen
    results = mp_pose.process(frame_rgb)
   
    # Verificar si se detectaron posturas corporales y dibujar los landmarks
    if results.pose_landmarks is not None:
        # Calcular ángulos específicos
        p23 = (int(results.pose_landmarks.landmark[23].x * image.shape[1]), int(results.pose_landmarks.landmark[23].y * image.shape[0]))
        p27 = (int(results.pose_landmarks.landmark[27].x * image.shape[1]), int(results.pose_landmarks.landmark[27].y * image.shape[0]))
        p28 = (int(results.pose_landmarks.landmark[28].x * image.shape[1]), int(results.pose_landmarks.landmark[28].y * image.shape[0]))
        p24 = (int(results.pose_landmarks.landmark[24].x * image.shape[1]), int(results.pose_landmarks.landmark[24].y * image.shape[0]))
        p26 = (int(results.pose_landmarks.landmark[26].x * image.shape[1]), int(results.pose_landmarks.landmark[26].y * image.shape[0]))
        p25 = (int(results.pose_landmarks.landmark[25].x * image.shape[1]), int(results.pose_landmarks.landmark[25].y * image.shape[0]))
        p11 = (int(results.pose_landmarks.landmark[11].x * image.shape[1]), int(results.pose_landmarks.landmark[11].y * image.shape[0]))
        p12 = (int(results.pose_landmarks.landmark[12].x * image.shape[1]), int(results.pose_landmarks.landmark[12].y * image.shape[0]))
        p13 = (int(results.pose_landmarks.landmark[13].x * image.shape[1]), int(results.pose_landmarks.landmark[13].y * image.shape[0]))
        p15 = (int(results.pose_landmarks.landmark[15].x * image.shape[1]), int(results.pose_landmarks.landmark[15].y * image.shape[0]))
        p30 = (int(results.pose_landmarks.landmark[30].x * image.shape[1]), int(results.pose_landmarks.landmark[30].y * image.shape[0]))
        p32 = (int(results.pose_landmarks.landmark[32].x * image.shape[1]), int(results.pose_landmarks.landmark[32].y * image.shape[0]))
        p29 = (int(results.pose_landmarks.landmark[29].x * image.shape[1]), int(results.pose_landmarks.landmark[29].y * image.shape[0]))
        p31 = (int(results.pose_landmarks.landmark[31].x * image.shape[1]), int(results.pose_landmarks.landmark[31].y * image.shape[0]))
        
        evaluar = form.get('switch')
        # evaluar=10
        anguloPiernas = int(calculate_angle(p25, p23, p26))
        anguloPiesDerecho = int(calculate_angle(p28, p30, p32))
        anguloPiesIzquierdo = int(calculate_angle(p31, p27, p29))
        anguloCaderaIzquierda=int(calculate_angle(p11,p23,p25))
        anguloCaderaDerecha=int(calculate_angle(p12,p24,p26))
        anguloCaderaBrazoIzquierdo=int(calculate_angle(p23,p11,p15))
        anguloAgarreBrazoIzquierdo=int(calculate_angle(p11,p13,p15))
        
        # Ruta de la carpeta donde se guardarán las imágenes
        carpeta_imagenes = "imagenes_posturas"
        
        # Crear la carpeta si no existe
        if not os.path.exists(carpeta_imagenes):
            os.makedirs(carpeta_imagenes)
        
        # Guardar la imagen con los puntos encontrados en la carpeta
        nombre_imagen = os.path.join(carpeta_imagenes, "imagen_con_puntos.jpg")
        cv2.imwrite(nombre_imagen, image)
        
        if(evaluar=="1"): #Evaluar hombros
            # print(p12[1])
            if(p12[1]<=p11[1]+30) and (p12[1]>=p11[1]-30):
                print("Hombros rectos")
                return jsonify({"status":True,"message":"Pose 1 correcta"}), 200
            else:
                print("Hombros INCORRECTOS")
                return jsonify({"status":False,"message":"Pose 1 incorrecta"}), 200
        elif (evaluar=="2"): #Evaluar piernas
            
            if(anguloPiernas>=35) and (anguloPiernas<=40):
                print("Piernas correcto")
                return jsonify({"status":True,"message":"Pose 2 correcta"}), 200
            else:
                print("Piernas incorrecto")
                return jsonify({"status":False,"message":"Pose 2 incorrecta"}), 200
        elif (evaluar=="3"): #Evaluar pies
            if(anguloPiernas>=35) and (anguloPiernas<=40) and (anguloPiesDerecho>=50) and (anguloPiesIzquierdo>=50):
                print("Pies exterior correcto")
                return jsonify({"status":True,"message":"Pose 3 correcta"}), 200
            else:
                print("Pies exterior incorrecto")
                return jsonify({"status":False,"message":"Pose 3 incorrecta"}), 200
        elif (evaluar=="4"): #Evaluar cadera izquierda
            if(anguloCaderaIzquierda>=140) and (anguloCaderaIzquierda<=160):
                print("CaderaIzquierda correcto")
                return jsonify({"status":True,"message":"Pose 4 correcta"}), 200
            else:
                print("CaderaIzquierda incorrecto")
                return jsonify({"status":False,"message":"Pose 4 incorrecta"}), 200
        elif (evaluar=="5"): #Evaluar cadera derecha
            if(anguloCaderaDerecha>=140) and (anguloCaderaDerecha<=160):
                print("Cadera Derecha correcto")
                return jsonify({"status":True,"message":"Pose 5 correcta"}), 200
            else:
                print("Cadera Derecha incorrecto")
                return jsonify({"status":False,"message":"Pose 5 incorrecta"}), 200
        elif (evaluar=="6"): #Parado recto
            if(p12[1]<=p11[1]+30) and (p12[1]>=p11[1]-30) and (anguloPiernas>=35) and (anguloPiernas<=40) and (anguloPiesDerecho>=50) and (anguloPiesIzquierdo>=50):
                print("Parado recto correcto")
                return jsonify({"status":True,"message":"Pose 6 correcta"}), 200
            else:
                print("Parado recto incorrecto")
                return jsonify({"status":False,"message":"Pose 6 incorrecta"}), 200
        elif (evaluar=="7"): #Levantar brazo
            if(anguloCaderaBrazoIzquierdo>=90 and anguloCaderaBrazoIzquierdo<=100):
                print("Levantar brazo izquierdo correcto")
                return jsonify({"status":True,"message":"Pose 7 correcta"}), 200
            else:
                print("Levantar brazo izquierdo incorrecto")
                return jsonify({"status":False,"message":"Pose 7 incorrecta"}), 200
        elif (evaluar=="8"): #Bajar brazo izquierdo
          if(anguloCaderaBrazoIzquierdo<=20)and(p12[1]<=p11[1]+30) and (p12[1]>=p11[1]-30) and (anguloPiernas>=35) and (anguloPiernas<=40) and (anguloPiesDerecho>=50) and (anguloPiesIzquierdo>=50):
                print("Bajar brazo izquierdo correcto")
                return jsonify({"status":True,"message":"Pose 8 correcta"}), 200
          else:
                print("Bajar brazo izquierdo incorrecto")
                return jsonify({"status":False,"message":"Pose 8 incorrecta"}), 200
        elif (evaluar=="9"): #Evaluar pies
          if(anguloCaderaBrazoIzquierdo>=90 and anguloCaderaBrazoIzquierdo<=100)and(p12[1]<=p11[1]+30) and (p12[1]>=p11[1]-30) and (anguloPiernas>=35) and (anguloPiernas<=40) and (anguloPiesDerecho>=50) and (anguloPiesIzquierdo>=50):
                print("Bajar brazo izquierdo correcto")
                return jsonify({"status":True,"message":"Pose 9 correcta"}), 200
          else:
                print("Bajar brazo izquierdo incorrecto")
                return jsonify({"status":False,"message":"Pose 9 incorrecta"}), 200
        else: #Evaluar postura
          if(anguloAgarreBrazoIzquierdo>=70 and anguloAgarreBrazoIzquierdo<=90) and (p12[1]<=p11[1]+30) and (p12[1]>=p11[1]-30) and (anguloPiernas>=35) and (anguloPiernas<=40) and (anguloPiesDerecho>=50) and (anguloPiesIzquierdo>=50):
                print("Postura correcta")
                return jsonify({"status":True,"message":"Pose 10 correcta"}), 200
          else:
                print("Postura incorrecto")
                return jsonify({"status":False,"message":"Pose 10 incorrecta"}), 200
        
    else:
        return jsonify({'error': 'No se detectaron posturas corporales en la imagen.'}), 400


def default_analyze_pose(results):
    return jsonify({'error': 'Número de función de análisis no válido.'}), 400

@app.route('/analyze_pose', methods=['POST'])
def analyze_pose_route():
    # Verificar si se proporcionó una imagen y un valor para el interruptor
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen.'}), 400
    if 'switch' not in request.form:
        return jsonify({'error': 'No se proporcionó ningún valor para el switch.'}), 400
    
    # Obtener la imagen y el valor del interruptor
    image_file = request.files['image']
    # switch_value = request.form['switch']
    
    # Llamar a la función de análisis adecuada según el valor del interruptor
    # if switch_value == '1':
    return analyze_practica1(image_file, request.form)
    # elif switch_value == '2':
        # return analyze_practica2(image_file, request.form)
    # Agrega más casos según sea necesario
    
    # Retornar un error si el valor del interruptor no es válido
    # return jsonify({'error': 'Valor de interruptor no válido.'}), 400


    

    

if __name__ == '__main__':
    app.run(debug=True)
