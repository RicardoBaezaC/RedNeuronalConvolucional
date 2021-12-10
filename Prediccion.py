#------- Importacion de librerías ---------
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array

ModeloDenso = 'C:/Users/Ricardo/PycharmProjects/RedesNeuronales/ClasificadorDenso.h5'
ModeloCNN = 'C:/Users/Ricardo/PycharmProjects/RedesNeuronales/ClasificadorCNN.h5'
ModeloCNN2 = 'C:/Users/Ricardo/PycharmProjects/RedesNeuronales/ClasificadorCNN2.h5'

#-------- Se leen las redes neuronales -----------
#Denso
Denso = tf.keras.models.load_model(ModeloDenso)
pesosDenso = Denso.get_weights();
Denso.set_weights(pesosDenso)

#CNN
CNN = tf.keras.models.load_model(ModeloCNN)
pesosCNN = CNN.get_weights();
CNN.set_weights(pesosCNN)

#CNN2
CNN2 = tf.keras.models.load_model(ModeloCNN2)
pesosCNN2 = CNN2.get_weights();
CNN2.set_weights(pesosCNN2)

#-------- Se realiza captura en video ------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200,200), interpolation = cv2.INTER_CUBIC)
    gray = np.array(gray).astype(float) / 255
    img = img_to_array(gray)
    img = np.expand_dims(img, axis = 0)

    prediccion = CNN2.predict(img)
    prediccion = prediccion[0]
    prediccion = prediccion[0]
    print(prediccion)

    if prediccion < 0.5:
        cv2.putText(frame, "Manzana", (200, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Pera", (200, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 2)

    cv2.imshow("Red Neuronal", frame)

    t = cv2.waitKey(1)
    if t ==27 or not cv2.getWindowProperty("Red Neuronal", cv2.WND_PROP_VISIBLE):
        break

cv2.destroyWindow("Red Neuronal")
cap.release()