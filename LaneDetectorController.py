import numpy as np
import cv2
from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import time
from datetime import datetime
import os

# Declaración de variables globales 
manual_steering = 0
steering_angle = 0  # Ángulo de dirección inicial
angle = 0.0  # Ángulo de dirección actual
speed = 60  # Velocidad inicial del robot


# Función para obtener imagen de la cámara
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# Función para procesamiento de la imagen
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

# Función para mostrar imagen en un display
def display_image(display, image):
    image_rgb = np.dstack((image, image, image))
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# Función para mostrar imagen con líneas de borde y ROI
def display_edge_screen(display, image, lines_detected, center_line, roi_coords):
    # Crearemos una copia de la imagen original
    image_copy = image.copy()

    # Se dibujan las líneas detectadas con Hough (en un azul oscuro)
    if lines_detected is not None:
        for line in lines_detected:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_copy, (x1, y1), (x2, y2), (80, 40, 50), 1) 
    
    # Se dibuja el centro de la línea amarilla principal
    if center_line[0] != -1:
        cx, cy = center_line
        cv2.circle(image_copy, (cx, cy), 5, (0, 255, 0), -1)  # Se utilizó color verde

    # Se resalta área del ROI con un rectángulo verde suave
    if roi_coords is not None:
        x, y, w, h = roi_coords
        overlay = image_copy.copy()
        alpha = 0.2  # Nivel de opacidad del color verde

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)  # Verde suave

        # Se mezcla imagen original y overlay para crear transparencia
        cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0, image_copy)

    # Buscamos que la imagen no se sobreponga, se utiliza un fondo blanco
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    # Se muestra la imagen final modificada en el display
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)
    return image_copy

# Se crea función para detectar línea amarilla del centro
def deteccion_linea_amarilla(image):

    # Convertimos la imagen a HSV para mejor segmentación de color
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definimos el rango para el color amarillo en HSV
    upper_color = np.array([40, 255, 255])
    lower_color = np.array([20, 100, 100])
    
    # Creamos una máscara para aislar el color amarillo
    yellow_mask = cv2.inRange(image_hsv, lower_color, upper_color)
    
    # Calculamos el centro de la línea amarilla
    center_line = cv2.moments(yellow_mask)
    
    # Analizamos momentos geométricos de la imagen
    if center_line['m00'] != 0:
        cx = int(center_line['m10'] / center_line['m00'])
        cy = int(center_line['m01'] / center_line['m00'])
    else:
        cx, cy = -1, -1  # Si no se detectó la línea
    
    # Se retorna la máscara y las coordenadas
    return yellow_mask, cx, cy

# Función para detección de líneas con la transformada de Hough
def deteccion_lineas(image):
    image_gray = greyscale_cv2(image)
    image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    edges_detected = cv2.Canny(image_blurred, 50, 150)
    lines_detected = cv2.HoughLinesP(edges_detected, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    return lines_detected, edges_detected

# Función para ajustar ángulo de dirección para el seguimiento de línea amarilla
def seguimiento_linea(cx, image_width):
    if cx == -1:
        return 0
    calculo_error = cx - image_width // 2
    steering_correction = calculo_error * 0.002
    steering_correction = np.clip(steering_correction, -0.5, 0.5)
    return steering_correction

# Funcion para actualizar ángulo de dirección
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    angle = wheel_angle

# Función para comprobar incremento del ángulo de dirección
def change_rotation_angle(inc):
    global manual_steering
    new_manual_steering = manual_steering + inc
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0:
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    if manual_steering == 0:
        print("Straight line")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print(f"Estamos girando {steering_angle} radianes a la {turn}")

# Se define la función principal
def main():
    global speed, angle, steering_angle 

    # Creamos instancia del robot
    robot = Car()
    driver = Driver()

    # Y obtenemos el time step del mundo
    timestep = int(robot.getBasicTimeStep())
    
    # Creamos la cámara
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Creamos displays que procesarán imágenes
    display_img = Display("display_image")  # Original
    display_edge = Display("display_edge")  # Imagen con bordes y ROI

    # Creamos la instancia del teclado
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Asignamos la velocidad inicial
    speed = 60 
    previous_key = None #Creamos variable para controlar cambios de velocidad

    while robot.step() != -1:
        # Aquí obtenemos la imagen de la cámara
        image = get_image(camera)

        # Ahora detectamos la línea amarilla
        yellow_mask, cx, cy = deteccion_linea_amarilla(image)

        # También detectamos líneas usando la transformada de Hough
        lines_detected, edges_detected = deteccion_lineas(image)

        # Se calcula el ROI en función de la línea amarilla del carril

        if cx != -1:
            # Definimos el ROI y su tamaño
            roi_ancho = 100
            roi_alto = 50
            roi_x = max(cx - roi_ancho // 2, 0)
            roi_y = max(cy - roi_alto // 2, 0)
            roi_coords = (roi_x, roi_y, roi_ancho, roi_alto)
        else:
            roi_coords = None

        # Imagen es mostrada y procesada
        grey_image = greyscale_cv2(image)
        display_image(display_img, grey_image)

        # También mostramos imagen con detección de líneas
        display_edge_screen(display_edge, image, lines_detected, (cx, cy), roi_coords)

        # Considerar que si detectamos la línea amarilla se ajustará el ángulo de dirección
        if cx != -1:
            steering_correction = seguimiento_linea(cx, image.shape[1])
            set_steering_angle(steering_correction)

        # Imprimimos el ángulo de dirección actual (en radianes)
        print(f"Ángulo de dirección: {steering_angle} radianes") 

        # Aquí leemos la entrada del teclado
        key = keyboard.getKey()

        # Solo se tomará acción si la tecla es diferente de la anterior
        if key != previous_key:
            if key == keyboard.UP:  # Subimos
                speed += 5.0  # Incremento de la velocidad
                print(f"Se aumenta velocidad a {speed}")
            elif key == keyboard.DOWN:  # Bajamos
                speed -= 5.0  # Se decrementa la velocidad
                print(f"Se reduce velocidad a {speed}")
            elif key == keyboard.RIGHT:  # Derecha
                change_rotation_angle(+1)
                print("Derecha")
            elif key == keyboard.LEFT:  # Izquierda
                change_rotation_angle(-1)
                print("Izquierda")
            elif key == ord('A'):  # Se almacena imagen procesada actual
                processed_image = display_edge_screen(display_edge, image, lines_detected, (cx, cy), roi_coords)
                current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
                file_name = current_datetime + "_edge.png"
                print("Se capturó imagen")
                cv2.imwrite(os.getcwd() + "/" + file_name, processed_image)

            # Se actualiza el último estado de la tecla presionada
            previous_key = key

        # También actualizamos el ángulo de dirección y la velocidad
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()
