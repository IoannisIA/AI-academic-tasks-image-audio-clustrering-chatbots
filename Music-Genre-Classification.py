# feature extractoring and preprocessing data
import librosa
import librosa.display

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import csv
import tensorflow as tf
# Preprocessing
from keras import Input, Model, Sequential, applications
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, SeparableConv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPooling2D, \
    GlobalAveragePooling2D
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn


genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
#Keras
import keras
import warnings
warnings.filterwarnings('ignore')

cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))



"""
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'./genres/{g}'):
        songname = f'./genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=1)
        n_mels = 128
        hop_length = 512
        n_fft = 2048

        S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
        #plt.colorbar(format='%+2.0f dB');


        #plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()
"""

batch_size=16

datagen_for_train = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen_for_train.flow_from_directory(
                            'img_data/train/',
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle='True',
                            subset='training') # set as training data

validation_generator = datagen_for_train.flow_from_directory(
                            'img_data/train/',
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle='True',
                            subset='validation') # set as validation data


datagen_for_test = ImageDataGenerator(rescale=1./255)

test_generator = datagen_for_test.flow_from_directory(
                            'img_data/test/',
                            batch_size=batch_size,
                            shuffle='True',
                            class_mode='categorical',) # set as validation data


x = test_generator.next()
print(x[0][1][2])


np.random.seed(23456)
tf.random.set_seed(123)

opt = Adam(lr=0.0001, decay=1e-5)


"""

cnn_model_2 = Sequential([
    Conv2D(16,3, padding='same', activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32,3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(64,3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])




"""




base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (256,256,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(10, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)


# Compiling our neural network
model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


es = EarlyStopping(patience=10)
chkpt = ModelCheckpoint(filepath='resnet_50', save_best_only=True, save_weights_only=True)

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    callbacks=[es, chkpt],
    epochs = 500)


"""



# Initiating an empty neural network
cnn_model = Sequential(name='cnn_1')

# Adding convolutional layer
cnn_model.add(Conv2D(filters=16,
                     kernel_size=(3,3),
                     activation='relu',
                     input_shape=(256,256,3)))

# Adding max pooling layer
cnn_model.add(MaxPooling2D(pool_size=(2,4)))

# Adding convolutional layer
cnn_model.add(Conv2D(filters=32,
                     kernel_size=(3,3),
                     activation='relu'))

# Adding max pooling layer
cnn_model.add(MaxPooling2D(pool_size=(2,4)))

# Adding a flattened layer to input our image data
cnn_model.add(Flatten())

# Adding a dense layer with 64 neurons
cnn_model.add(Dense(64, activation='relu'))

# Adding a dropout layer for regularization
cnn_model.add(Dropout(0.4))

# Adding an output layer
cnn_model.add(Dense(10, activation='softmax'))



# Compiling our neural network
cnn_model_2.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])



history = cnn_model_2.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = 500)

"""


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




# Load the model weights
model.load_weights("resnet_50")





test_loss, test_score =  model.evaluate(test_generator)

print("Loss on test set: ", test_loss)
print("Accuracy on test set: ", test_score)



#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=genres))



df_cm = pd.DataFrame(confusion_matrix(test_generator.classes, y_pred), index = [i for i in genres],
                  columns = [i for i in genres])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

plt.show()
"""
#this is the second try with text features

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()


file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'./genres/{g}'):
        songname = f'./genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


data = pd.read_csv('data.csv')

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)


genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)


scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


from keras import models
from keras import layers


x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]


model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

opt = Adam(lr=0.0001, decay=1e-5)
es = EarlyStopping(patience=15)
chkpt = ModelCheckpoint(filepath='best_model_todatexxxx', save_best_only=True, save_weights_only=True)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train,
          partial_y_train,
          epochs=500,
          batch_size=64,
          callbacks=[es, chkpt],
          validation_data=(x_val, y_val))


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()




# Load the model weights
model.load_weights("best_model_todatexxxx")



test_loss, test_score =  model.evaluate(X_test, y_test)

print("Loss on test set: ", test_loss)
print("Accuracy on test set: ", test_score)



#Confution Matrix and Classification Report
Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

print('Classification Report')
print(classification_report(y_test, y_pred, target_names=genres))



df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), index = [i for i in genres],
                  columns = [i for i in genres])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

plt.show()
"""