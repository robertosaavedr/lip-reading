from sequentia.classifiers import GMMHMM, HMMClassifier
import cv2 as cv
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from keras.preprocessing import sequence
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import SimpleRNN
import keras
from sklearn.utils import resample

from itertools import compress

labels = pd.DataFrame(columns=['person' , 'sentence' , 'attemp' , 'frame'])
in_matrix = None
labels_matrix = None

for person in os.listdir("lip_reading/mouths"):
    for sentence in os.listdir("lip_reading/mouths/%s" % person):
        for attemp in os.listdir("lip_reading/mouths/%s/%s" % (person, sentence)):
            for frame in sorted(os.listdir("lip_reading/mouths/%s/%s/%s" % (person, sentence, attemp))):
                im = cv.imread("lip_reading/mouths/%s/%s/%s/%s" % (person, sentence, attemp, frame), cv.IMREAD_GRAYSCALE)
                #sizes.append(im.shape) 
                im = cv.resize(im, [32, 16], interpolation= cv.INTER_CUBIC)
                vec = im.reshape(32*16)
                size = np.array([len(os.listdir("lip_reading/mouths/%s/%s/%s" % (person, sentence, attemp))), person, int(sentence), int(attemp), frame])
                try:
                    in_matrix = np.vstack((in_matrix, vec))
                    labels_matrix = np.vstack((labels_matrix, size))
                except:
                    in_matrix = vec
                    labels_matrix = size

# dividir en test y entrenamineto
indexes_test = np.logical_or(labels_matrix[:, 1] == "F10", labels_matrix[:, 1] == "M02")
indexes_validation = labels_matrix[:, 1] == "F11"
indexes_train = np.logical_and(indexes_test == False, indexes_validation == False)

X_train, X_validation, X_test = in_matrix[indexes_train, :], in_matrix[indexes_validation, :], in_matrix[indexes_test, :]
y_train_total, y_validation_total, y_test_total = labels_matrix[indexes_train, :], labels_matrix[indexes_validation, :], labels_matrix[indexes_test, :]
# PCA
pca = PCA(n_components=10)
pca.fit(X_train)

X_train_transformed = pca.transform(X_train)
X_test_transformed = pca.transform(X_test)
X_validation_transformed = pca.transform(X_validation)

## TRAIN
indexes = y_train_total[:, 0].astype(int)
order = []
aux = 0
for i in range(1200):
    order.extend([indexes[aux]])
    aux = aux + indexes[aux]

cum = 0
lower = 0
seq = []
y_train = np.array([])
X_train_3d = list()

for i in order:
    cum = cum + i
    X_train_3d.append(X_train_transformed[lower:cum, :].reshape(i, 10))
    y_train = np.append(y_train, y_train_total[lower, 2])
    lower = cum


## TEST 
indexes = y_test_total[:, 0].astype(int)
order = []
aux = 0
for i in range(200):
    order.extend([indexes[aux]])
    aux = aux + indexes[aux]

cum = 0
lower = 0
seq = []
y_test = np.array([])
X_test_3d = list()

for i in order:
    cum = cum + i
    X_test_3d.append(X_test_transformed[lower:cum, :].reshape(i, 10))
    y_test = np.append(y_test, y_test_total[lower, 2])
    lower = cum

# validation data
indexes = y_validation_total[:, 0].astype(int)
order = []
aux = 0
for i in range(100):
    order.extend([indexes[aux]])
    aux = aux + indexes[aux]

cum = 0
lower = 0
seq = []
y_validation = np.array([])
X_validation_3d = list()

for i in order:
    cum = cum + i
    X_validation_3d.append(X_validation_transformed[lower:cum, :].reshape(i, 10))
    y_validation = np.append(y_validation, y_validation_total[lower, 2])
    lower = cum

# training with hmm gmm

labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_train = y_train.astype(int)
y_test = y_test.astype(int)
y_validation = y_validation.astype(int)

hmms = []


for i, label in enumerate(labels):
    hmm = GMMHMM(label = label, n_states=6, n_components=9, topology='left-right')
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(list(compress(X_train_3d, y_train==label)))
    hmms.append(hmm)

clf = HMMClassifier()
clf.fit(hmms)
# using validation set to tune hyperparameters/topology of hmm
# predictions = clf.predict(X_validation_3d)
accuracy, confusion = clf.evaluate(X_validation_3d, y_validation)
accuracy, confusion = clf.evaluate(X_train_3d, y_train)

