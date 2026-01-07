import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
import keras

# ----------------------------------------------------------------
# PASO 1: Cargar las imágenes de sportsimages
# ----------------------------------------------------------------

dirname = os.path.join(os.getcwd(), 'sportimages')
imgpath = dirname + os.sep

images = []
directories = []
dircount = []
prevRoot = ''
cant = 0

print("leyendo imagenes de ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("(jpg|jpeg|png|bmp|tiff)$", filename):
            cant = cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            images.append(image)
            b = "Leyendo... " + str(cant)
            print (b, end="\r")

    # --- Esta es la segunda parte de la imagen ---
    if prevRoot != root:
        print(root, cant)
        prevRoot = root
        directories.append(root)
        dircount.append(cant)
        cant = 0

dircount.append(cant)

dircount = dircount[1:]
#dircount[0] = dircount[0] + 1
print('Directorios leidos: ', len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:', sum(dircount))

# ----------------------------------------------------------------
# PASO 2: Crear etiquetas y arrays finales
# ----------------------------------------------------------------
labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("Cantidad etiquetas creadas: ", len(labels))

deportes=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    
    # --- Continuación del bucle (de la segunda imagen) ---
    deportes.append(name[len(name)-1])
    indice=indice+1

# --- Resto del código de la segunda imagen ---
y = np.array(labels)
X = np.array(images, dtype=np.uint8) #convierto de lista a numpy

# Find the unique numbers from the train labels
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# ----------------------------------------------------------------
# PASO 3: Dividir y preprocesar los datos 
# ----------------------------------------------------------------
# Mezclar todo y crear los grupos de entrenamiento y testing
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2)
print('\nTraining data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# Normalizar los datos de imagen (píxeles de 0-255 a 0-1)
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Convertir las etiquetas a formato one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Mostrar el cambio
print('\nOriginal label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

# Crear un set de validación a partir del set de entrenamiento
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print('\nFinal shapes:')
print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

# ----------------------------------------------------------------
# PASO 4: Crear y entrenar el modelo
# ----------------------------------------------------------------

INIT_LR = 1e-3
epochs = 6
batch_size = 64

sport_model = Sequential()
sport_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(21, 28, 3)))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D((2, 2), padding='same'))
sport_model.add(Dropout(0.5))

sport_model.add(Flatten())
sport_model.add(Dense(32, activation='linear'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(Dropout(0.5))
sport_model.add(Dense(nClasses, activation='softmax'))

sport_model.summary()

sport_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adagrad(learning_rate=INIT_LR),
    metrics=['accuracy']
)