from subprocess import check_output
import cv2 as cv
import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.preprocessing
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
import keras
from keras import Sequential
from keras.layers import Conv3D, MaxPooling3D, LSTM, Reshape, Flatten, Dense
from keras.regularizers import l2

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


# dividir en test y entrenamiento
indexes_test = np.logical_or(labels_matrix[:, 1] == "F10", labels_matrix[:, 1] == "M02")
indexes_validation = labels_matrix[:, 1] == "F11"
indexes_train = np.logical_and(indexes_test == False, indexes_validation == False)

X_train, X_validation, X_test = in_matrix[indexes_train, :], in_matrix[indexes_validation, :], in_matrix[indexes_test, :]
y_train, y_validation, y_test = labels_matrix[indexes_train, :], labels_matrix[indexes_validation, :], labels_matrix[indexes_test, :]


#entrenamiento
indexes = y_train[:, 0].astype(int)
order = []
aux = 0

for i in range(1100):
    order.extend([indexes[aux]])
    aux = aux + indexes[aux]

cum = 0
lower = 0
seq = []
#seq = np.array([])
y = np.array([])
for i in order:
    cum = cum + i
    seq.append(X_train[lower:cum, :])
    y = np.append(y, y_train[lower, 2])
    lower = cum

y_train = y.astype(int)
X_train = sequence.pad_sequences(seq)
# test
indexes = y_test[:, 0].astype(int)
order = []
aux = 0

for i in range(200):
    order.extend([indexes[aux]])
    aux = aux + indexes[aux]

cum = 0
lower = 0
seq = []
#seq = np.array([])
y = np.array([])
for i in order:
    cum = cum + i
    seq.append(X_test[lower:cum, :])
    y = np.append(y, y_test[lower, 2])
    lower = cum

y_test = y.astype(int)
X_test = sequence.pad_sequences(seq, maxlen=27)
# validation 

indexes = y_validation[:, 0].astype(int)
order = []
aux = 0

for i in range(100):
    order.extend([indexes[aux]])
    aux = aux + indexes[aux]

cum = 0
lower = 0
seq = []
#seq = np.array([])
y = np.array([])
for i in order:
    cum = cum + i
    seq.append(X_validation[lower:cum, :])
    y = np.append(y, y_validation[lower, 2])
    lower = cum

y_validation = y.astype(int)
X_validation = sequence.pad_sequences(seq, maxlen=27)


X_train = X_train.reshape(1100, 27, 27, 13, 1)
X_test = X_test.reshape(200, 27, 27, 13, 1)
X_validation = X_validation.reshape(100, 27, 27, 13, 1)

encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.fit_transform(y_test.reshape(-1, 1))
y_validation = encoder.fit_transform(y_validation.reshape(-1, 1))

X_train = X_train/255
X_test = X_test/255
X_validation = X_validation/255

################# end-to-end nn #1
model = Sequential(
    [
        Conv3D(16, (3, 3, 3), input_shape=(27, 27, 13, 1), 
        activation='relu', padding='valid', kernel_regularizer=l2(0.01), 
        bias_regularizer=l2(0.01)),
        MaxPooling3D(pool_size=(2, 2, 2), strides=2),
        Reshape((16, 12*5*12)),
        LSTM(32, return_sequences=True),
        Flatten(),
        Dense(2048,  activation='relu'),
        Dense(1024,  activation='relu'),
        Dense(10, activation='softmax')
    ]
)


model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

checkpoint_filepath = 'phrases_checkpoint.h5'
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                    batch_size=32, epochs=20, callbacks=[early_stopping, model_checkpoint])



# plots to check loss and accuracy in validation and train
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.show()
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.show()


model.load_weights("phrases_checkpoint.h5")
model.evaluate(X_test, y_test)
model.evaluate(X_validation, y_validation) 
model.evaluate(X_train, y_train)

# confusion matrix
testing = encoder.inverse_transform(y_test)

sklearn.metrics.confusion_matrix(testing, model.predict(X_test).argmax(axis=1)+1)
