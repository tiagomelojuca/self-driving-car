import os
import random

import numpy as np
import pandas as pd
import cv2 as cv

import matplotlib.pyplot as plt
import matplotlib.image as iplt

from imgaug import augmenters as ai

from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------------------------------------------------------------------
CSV_FILE  = "driving_log.csv"
BASE_PATH = "C:\\Users\\tiago\\OneDrive\\Área de Trabalho\\TMP\\_unifor\\si\\av3\\src\\datasets"
ANG_MIN   = -25
ANG_MAX   =  25
# ---------------------------------------------------------------------------------------

def getDatasetCSV(dataset): # DataSet
    # dataset   = "pista_mix_ida_kb"
    csv_path  = os.path.join(os.path.join(BASE_PATH, dataset), CSV_FILE)
    return csv_path

# ---------------------------------------------------------------------------------------

def getDataframe(dataset): # DataFrame
    _names = ["centro", "esquerda", "direita", "angulo", "aceleracao", "freios", "velocidade"]
    return pd.read_csv(getDatasetCSV(dataset), names=_names)

# ---------------------------------------------------------------------------------------

def queryNumBins():
    return int(abs(ANG_MIN) + 1 + abs(ANG_MAX))

# ---------------------------------------------------------------------------------------

def queryCenterBins():
    return int((abs(ANG_MIN) + abs(ANG_MAX)) / 2.0)

# ---------------------------------------------------------------------------------------

def visualizeData(dataframe, lim):
    nBins = queryNumBins()
    hist, bins = np.histogram(dataframe["angulo"], nBins)
    centro = (bins[:-1] + bins[1:]) * 0.5
    plt.bar(centro, hist, width = 0.06)
    plt.xlim((-1, 1))
    plt.plot((-1, 1), (lim, lim))
    plt.show()

# ---------------------------------------------------------------------------------------

def selectData(dataframe, lim):
    numBins = queryNumBins()
    cenBins = queryCenterBins()
    _, bins = np.histogram(dataframe["angulo"], numBins)

    arr = []
    for i in range(len(dataframe["angulo"])):
        if(dataframe["angulo"][i] >= bins[cenBins] and dataframe["angulo"][i] <= bins[cenBins + 1]):
            arr.append(i)
    seed = np.random.permutation(len(arr))
    arr = np.asarray(arr)
    arr = arr[seed]
    arr = arr[lim:]
    dataframe.drop(dataframe.index[arr], inplace = True)

# ---------------------------------------------------------------------------------------

def parseDataframe(dataframe):
    imagens = []
    angulos = []

    for i in range(len(dataframe)):
        indexData = dataframe.iloc[i]
        imagens.append(indexData["centro"])
        angulos.append(indexData["angulo"])
    imagens = np.asarray(imagens)
    angulos = np.asarray(angulos)

    return imagens, angulos

# ---------------------------------------------------------------------------------------

def prepareDataset(imagens, angulos):
    p = np.random.permutation(len(imagens))
    _imagens = imagens[p]
    _angulos = angulos[p]
    xTrain = _imagens[0 : int(len(_imagens) * 0.8)]
    yTrain = _angulos[0 : int(len(_angulos) * 0.8)]
    xTest  = _imagens[int(len(_imagens) * 0.8) : len(_imagens)]
    yTest  = _angulos[int(len(_angulos) * 0.8) : len(_angulos)]

    return xTrain, yTrain, xTest, yTest

# ---------------------------------------------------------------------------------------

def preprocess(img):
    img = img[60 : 135, :, :]
    img = cv.cvtColor(img, cv.COLOR_RGB2YUV)
    img = cv.GaussianBlur(img, (3, 3), 0)
    img = cv.resize(img, (200, 66))
    img = img / 255

    # cv.imshow('image', img)
    # cv.waitKey(0)

    return img

# ---------------------------------------------------------------------------------------

def augmentImage(imgPath, ang):
    img = iplt.imread(imgPath)

    if np.random.rand() > 0.5:
        translation = ai.Affine(translate_percent = { "x": (-0.1, 0.1), "y": (-0.1, 0.1) })
        img = translation.augment_image(img)

    if np.random.rand() > 0.5:
        zoom = ai.Affine(scale = (1, 1.2))
        img = zoom.augment_image(img)
    
    if np.random.rand() > 0.5:
        brightness = ai.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    
    if np.random.rand() > 0.5:
        img = cv.flip(img, 1)
        ang = -ang

    return img, ang

# ---------------------------------------------------------------------------------------

def batchGenerator(imgPaths, angles, size, shouldAugmentImage):
    while True:
        imgBatch = []
        angBatch = []
        for i in range(size):
            idx = random.randint(0, len(imgPaths) - 1)
            if shouldAugmentImage:
                img, ang = augmentImage(imgPaths[idx], angles[idx])
            else:
                img = iplt.imread(imgPaths[idx])
                ang = angles[idx]
            img = preprocess(img)
            imgBatch.append(img)
            angBatch.append(ang)
        yield np.asarray(imgBatch), np.asarray(angBatch)

# ---------------------------------------------------------------------------------------

def createModel():
    model = Sequential()
    model.add(Convolution2D(16, (2,2), input_shape=(66, 200, 3), activation="elu"))
    model.add(MaxPooling2D((2, 2), (1, 3)))
    model.add(Convolution2D(24, (3, 3), activation="elu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(32, (5, 5), activation="elu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, (5, 5), activation="elu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))
    model.add(Dropout(rate = 0.2))
    model.compile(Adam(learning_rate = 0.0001), loss = "mse")

    model.summary()

    return model

# ---------------------------------------------------------------------------------------

def exec():
    ES = EarlyStopping(
        monitor = "val_loss",
        mode = "min",
        verbose = 1,
        patience = 5
    )

    MC = ModelCheckpoint(
        "TheBest.h5",
        monitor = "val_loss",
        mode = "min",
        verbose = 1,
        save_best_only = True
    )

    lim = 300
    dataframe = getDataframe("pista2_volta_m")
    selectData(dataframe, lim)
    imagens, angulos = parseDataframe(dataframe)
    xTrain, yTrain, xValid, yValid = prepareDataset(imagens, angulos)

    model = createModel()

    qtdTrain = 200
    qtdValid = 200
    hist = model.fit(
        batchGenerator(xTrain, yTrain, qtdTrain, True),
        steps_per_epoch = 200,
        epochs = 100,
        validation_data = batchGenerator(xValid, yValid, qtdValid, False),
        callbacks = [ES, MC],
        validation_steps = 200
    )

    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.legend(["treino", "validacao"])
    plt.show()

# ---------------------------------------------------------------------------------------

exec()

# ---------------------------------------------------------------------------------------
