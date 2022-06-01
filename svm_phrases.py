import cv2 as cv
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from keras.preprocessing import sequence
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from sklearn.utils import resample
import sklearn.metrics

labels = pd.DataFrame(columns=['person' , 'sentence' , 'attemp' , 'frame'])
in_matrix = None
labels_matrix = None

for person in os.listdir("lip_reading/mouths"):
    for sentence in os.listdir("lip_reading/mouths/%s" % person):
        for attemp in os.listdir("lip_reading/mouths/%s/%s" % (person, sentence)):
            for frame in sorted(os.listdir("lip_reading/mouths/%s/%s/%s" % (person, sentence, attemp))):
                im = cv.imread("lip_reading/mouths/%s/%s/%s/%s" % (person, sentence, attemp, frame), cv.IMREAD_GRAYSCALE)
                #sizes.append(im.shape) 
                im = cv.resize(im, [27, 13], interpolation= cv.INTER_CUBIC)
                vec = im.reshape(27*13)
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
y_train, y_validation, y_test = labels_matrix[indexes_train, :], labels_matrix[indexes_validation, :], labels_matrix[indexes_test, :]


# PCA
pca = PCA(n_components=10)
pca.fit(X_train)
# selection the number of components
plt.plot(np.arange(pca.n_components_) + 1, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
pca.explained_variance_ratio_.cumsum()

X_train_transformed = pca.transform(X_train)
X_test_transformed = pca.transform(X_test)
X_validation_transformed = pca.transform(X_validation)
# one row per utterance
## TRAIN
indexes = y_train[:, 0].astype(int)
order = []
aux = 0
for i in range(1200):
    order.extend([indexes[aux]])
    aux = aux + indexes[aux]

cum = 0
lower = 0
seq = []
y = np.array([])
for i in order:
    cum = cum + i
    seq.append(X_train_transformed[lower:cum, :].ravel().tolist())
    y = np.append(y, y_train[lower, 2])
    lower = cum

## TEST 

indexes = y_test[:, 0].astype(int)
order = []
aux = 0
for i in range(200): # 100 because we have 1 person in test set, 1 x 10 words x 10 repetetitions
    order.extend([indexes[aux]])
    aux = aux + indexes[aux]

cum = 0
lower = 0
seq_test = []
y_test2 = np.array([])
for i in order:
    cum = cum + i
    seq_test.append(X_test_transformed[lower:cum, :].ravel().tolist())
    y_test2 = np.append(y_test2, y_test[lower, 2])
    lower = cum

## Validation

indexes = y_validation[:, 0].astype(int)
order = []
aux = 0
for i in range(100): # 100 because we have 1 person in test set, 1 x 10 words x 10 repetetitions
    order.extend([indexes[aux]])
    aux = aux + indexes[aux]

cum = 0
lower = 0
seq_validation = []
y_validation2 = np.array([])
for i in order:
    cum = cum + i
    seq_validation.append(X_validation_transformed[lower:cum, :].ravel().tolist())
    y_validation2 = np.append(y_validation2, y_validation[lower, 2])
    lower = cum


# padding
seq_transformed = sequence.pad_sequences(seq, 270)
seq_transformed_test = sequence.pad_sequences(seq_test, 270)
seq_transformed_validation = sequence.pad_sequences(seq_validation, 270)

seq_transformed.shape
seq_transformed_test.shape
seq_transformed_validation.shape

#SVM
y = y.astype(int)
y_test2 = y_test2.astype(int)
y_validation2 = y_validation2.astype(int)

clf = make_pipeline(StandardScaler(), SVC(kernel="linear"))
clf.fit(seq_transformed, y)

clf.score(seq_transformed, y)
clf.score(seq_transformed_validation, y_validation2)

clf.score(seq_transformed_test, y_test2)

sklearn.metrics.confusion_matrix(y_test2, clf.predict(seq_transformed_test))
sklearn.metrics.confusion_matrix(y, clf.predict(seq_transformed))

