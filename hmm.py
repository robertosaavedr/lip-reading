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

for person in os.listdir("lip_reading/words"):
    for sentence in os.listdir("lip_reading/words/%s" % person):
        for attemp in os.listdir("lip_reading/words/%s/%s" % (person, sentence)):
            for frame in sorted(os.listdir("lip_reading/words/%s/%s/%s" % (person, sentence, attemp))):
                im = cv.imread("lip_reading/words/%s/%s/%s/%s" % (person, sentence, attemp, frame), cv.IMREAD_GRAYSCALE)
                #sizes.append(im.shape) 
                im = cv.resize(im, [32, 16], interpolation= cv.INTER_CUBIC)
                vec = im.reshape(32*16)
                size = np.array([len(os.listdir("lip_reading/words/%s/%s/%s" % (person, sentence, attemp))), person, int(sentence), int(attemp), frame])
                try:
                    in_matrix = np.vstack((in_matrix, vec))
                    labels_matrix = np.vstack((labels_matrix, size))
                except:
                    in_matrix = vec
                    labels_matrix = size

# dividir en test y entrenamineto
indexes_test = labels_matrix[:, 1] == "F11"
indexes_validation = labels_matrix[:, 1] == "NA"
indexes_train = np.logical_and(indexes_test == False, indexes_validation == False)

X_train, _, X_test = in_matrix[indexes_train, :], in_matrix[indexes_validation, :], in_matrix[indexes_test, :]
y_train_total, _, y_test_total = labels_matrix[indexes_train, :], labels_matrix[indexes_validation, :], labels_matrix[indexes_test, :]
# PCA
pca = PCA(n_components=15)
pca.fit(X_train)

X_train_transformed = pca.transform(X_train)
X_test_transformed = pca.transform(X_test)

## TRAIN
indexes = y_train_total[:, 0].astype(int)
order = []
aux = 0
for i in range(1400):
    order.extend([indexes[aux]])
    aux = aux + indexes[aux]

cum = 0
lower = 0
seq = []
y = np.array([])
X_train_3d = list()

for i in order:
    cum = cum + i
    X_train_3d.append(X_train_transformed[lower:cum, :].reshape(i, 15))
    y = np.append(y, y_train_total[lower, 2])
    lower = cum


## TEST 
indexes = y_test_total[:, 0].astype(int)
order = []
aux = 0
for i in range(100):
    order.extend([indexes[aux]])
    aux = aux + indexes[aux]

cum = 0
lower = 0
seq = []
y_test = np.array([])
X_test_3d = list()

for i in order:
    cum = cum + i
    X_test_3d.append(X_test_transformed[lower:cum, :].reshape(i, 15))
    y_test = np.append(y_test, y_test_total[lower, 2])
    lower = cum


labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = y.astype(int)
y_test = y_test.astype(int)
hmms = []


for i, label in enumerate(labels):
    hmm = GMMHMM(label = label, n_states=8, n_components=15, topology='left-right')
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(list(compress(X_train_3d, y==label)))
    hmms.append(hmm)

clf = HMMClassifier()
clf.fit(hmms)
predictions = clf.predict(X_test_3d)
accuracy, confusion = clf.evaluate(X_test_3d, y_test)
accuracy, confusion = clf.evaluate(X_train_3d, y)

#knn
import sequentia

clf = sequentia.KNNClassifier(k=2, classes=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
clf.fit(list(X_train_3d[i, :, :] for i in range(1200)), y)

preds = clf.predict(list(X_test_3d[i, :, :] for i in range(200)))
sum(preds == y_test)