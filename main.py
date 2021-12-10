#------- Importacion de librerías ---------
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#------- Direcciones de los datasets ---------
entrenamiento = 'C:/Users/Ricardo/PycharmProjects/RedesNeuronales/Dataset/Entrenamiento'
validacion = 'C:/Users/Ricardo/PycharmProjects/RedesNeuronales/Dataset/Validacion'


listaTrain = os.listdir(entrenamiento)
listaTest = os.listdir(validacion)

#-------- Establecemos parámetros ----------
ancho, alto = 200,200
#Listas de entrenamiento
etiquetas = []
fotos = []
datos_train = []
con = 0
#Lista de vallidación
etiquetas2 = []
fotos2 = []
datos_valid2 = []
con2 = 0

#--------- Extraemos en una lista las fotos y entra las etiquetas -----------
#Entrenamiento
for nameDir in listaTrain:
    nombre = entrenamiento + '/' + nameDir

    for fileName in os.listdir(nombre):
        etiquetas.append(con)
        img = cv2.imread(nombre + '/' + fileName, 0)
        img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC)
        img = img.reshape(ancho, alto, 1)
        datos_train.append([img, con])
        fotos.append(img)
    con = con + 1

#Validacion
for nameDir2 in listaTest:
    nombre2 = validacion + '/' + nameDir2

    for fileName2 in os.listdir(nombre2):
        etiquetas2.append(con2)
        img2 = cv2.imread(nombre2 + '/' + fileName2, 0)
        img2 = cv2.resize(img2, (ancho, alto), interpolation=cv2.INTER_CUBIC)
        img2 = img2.reshape(ancho, alto, 1)
        datos_train.append([img2, con2])
        fotos2.append(img2)
    con2 = con2 + 1

#----------- Se normalizan las imagenes ----------
fotos = np.array(fotos).astype(float) / 255
print(fotos.shape)
fotos2 = np.array(fotos2).astype(float) / 255
print(fotos2.shape)
#Se pasan las listas a Array
etiquetas = np.array(etiquetas)
etiquetas2 = np.array(etiquetas2)

imgTrainGen = ImageDataGenerator(
    rotation_range = 50,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    shear_range = 15,
    zoom_range = [0.5, 1.5],
    vertical_flip = True,
    horizontal_flip = True
)

imgTrainGen.fit(fotos)
plt.figure(figsize=(20,8))
for imagen, etiqueta in imgTrainGen.flow(fotos, etiquetas, batch_size = 10, shuffle = False):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[i], cmap='gray')
    plt.show()
    break

imgTrain = imgTrainGen.flow(fotos, etiquetas, batch_size=32)

#---------- Estructura de la Red Neuronal Convolucional ----------
#Modelo con Capas Densas
ModeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (200,200,1)),
    tf.keras.layers.Dense(150, activation = 'relu'),
    tf.keras.layers.Dense(150, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
])

#Modelo con Capas Convolucionales
ModeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (200,200,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    #Capas Densas de Clasificación
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

ModeloCNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (200,200,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    #Capas Densas de Clasificación
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

#-------- Se compilan los modelos, Se agrega el optimizador y la función de perdida --------
ModeloDenso.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

ModeloCNN.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

ModeloCNN2.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

#-------- Entrenamiento Modelo Denso ----------
BoardDenso = TensorBoard(log_dir='C:/Users/Ricardo/PycharmProjects/RedesNeuronales')
ModeloDenso.fit(imgTrain, batch_size = 32, validation_data = (fotos2,etiquetas2),
                epochs = 100, callbacks = [BoardDenso], steps_per_epoch = int(np.ceil(len(fotos) / float(32))),
                validation_steps = int(np.ceil(len(fotos2) / float(32))))
#-------- Se guarda el modelo ----------
ModeloDenso.save('ClasificadorDenso.h5')
ModeloDenso.save_weights('pesoDenso.h5')
print('Terminamos modelo denso')

#-------- Entrenamiento Modelo CNN sin DO ----------
BoardCNN = TensorBoard(log_dir='C:/Users/Ricardo/PycharmProjects/RedesNeuronales')
ModeloCNN.fit(imgTrain, batch_size = 32, validation_data = (fotos2,etiquetas2),
                epochs = 100, callbacks = [BoardCNN], steps_per_epoch = int(np.ceil(len(fotos) / float(32))),
                validation_steps = int(np.ceil(len(fotos2) / float(32))))
#-------- Se guarda el modelo ----------
ModeloCNN.save('ClasificadorCNN.h5')
ModeloCNN.save_weights('pesoCNN.h5')
print('Terminamos modelo CNN 1')

#-------- Entrenamiento Modelo CNN con DO ----------
BoardCNN2 = TensorBoard(log_dir='C:/Users/Ricardo/PycharmProjects/RedesNeuronales')
ModeloCNN2.fit(imgTrain, batch_size = 32, validation_data = (fotos2,etiquetas2),
                epochs = 100, callbacks = [BoardCNN2], steps_per_epoch = int(np.ceil(len(fotos) / float(32))),
                validation_steps = int(np.ceil(len(fotos2) / float(32))))
#-------- Se guarda el modelo ----------
ModeloCNN2.save('ClasificadorCNN2.h5')
ModeloCNN2.save_weights('pesoCNN2.h5')
print('Terminamos modelo CNN 2')