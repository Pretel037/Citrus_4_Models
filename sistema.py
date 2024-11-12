import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
from datetime import datetime


# Cargar modelos TFLite con rutas relativas
model_paths = {
    "DenseNet121": "models/densetnet_121.tflite",
    "Modelo 2": "models/GoogleNetLite.tflite",
    "Modelo 3": "models/densetnet_121.tflite",
    "Modelo 4": "models/citrus_modelLite.tflite"
}

# Crear un diccionario para los intérpretes de TFLite
interpreters = {name: tf.lite.Interpreter(model_path=path) for name, path in model_paths.items()}
for interpreter in interpreters.values():
    interpreter.allocate_tensors()
# Crear directorio para guardar imágenes
image_folder = "imagenes"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Función para predecir usando el modelo seleccionado
def image_prediction(image, interpreter):
    # Preprocesar la imagen
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (265, 265))
    image = np.expand_dims(image, axis=0).astype(np.float32)

    # Ejecutar predicción
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_class = np.argmax(pred)

    # Mapear a etiquetas de enfermedades
    labels = ["Mancha Negra", "Cancro", "Enverdecimiento", "Saludable"]
    return labels[pred_class]

# Interfaz de Streamlit
st.title("CITRUS")

# Seleccionar modelo
selected_model_name = st.selectbox("Selecciona el modelo para predicción", list(interpreters.keys()))
selected_interpreter = interpreters[selected_model_name]

# Capturar imagen desde la cámara
image = st.camera_input("Captura una imagen para analizar")

if image:
    # Guardar imagen en la carpeta
    image_file = Image.open(image)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(image_folder, f"captura_{timestamp}.png")
    image_file.save(image_path)
    st.write(f"Imagen guardada en {image_path}")

    # Predecir con el modelo seleccionado
    result = image_prediction(image_file, selected_interpreter)
    st.write(f"El modelo predice que la imagen tiene: {result}")