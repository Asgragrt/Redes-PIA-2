# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

# %%
def load_audio_16k(filename, mono=False):
    audio = tfio.audio.AudioIOTensor(filename, dtype=tf.float32)

    sample_rate = tf.cast(audio.rate, dtype=tf.int64)

    audio = audio[:]

    if mono:
        audio = audio[:,0]
    
    audio = tfio.audio.resample(audio, rate_in=sample_rate, rate_out=16000)

    return audio

# %%
def preprocess(file_path, label):
    audio = load_audio_16k(file_path)
    #Using two channels
    step1 = len(audio[:,0]) // 16
    step2 = len(audio[:,0]) // 33

    labels = np.ones((30,1), dtype=np.float32) * label


    steps1 = np.arange(15)*step1
    steps2 = np.arange(start=1, stop=30, step=2)*step2
    audioWindow = np.arange(10000, 58000).reshape(-1,1)
    audioWindow1 = audioWindow + steps1
    audioWindow2 = audioWindow + steps2


    audio = tf.transpose(tf.concat([tf.gather(audio[:,0], audioWindow1), tf.gather(audio[:,1], audioWindow2)], 1))

    spectrograms = tfio.audio.spectrogram(audio, nfft=512, window=960, stride=240)
    spectrograms = tfio.audio.melscale(spectrograms, rate=16000, mels=100, fmin=0, fmax=4000)
    spectrograms = tfio.audio.dbscale(spectrograms, top_db=70)

    spectrograms = tf.expand_dims(spectrograms, axis=3)
    
    return spectrograms, labels

# %%
b, _ = preprocess(os.path.join('/kaggle/input/musicbpm','data', 'Training', '94_Zankoku na Tenshi no These.mp3'), 94)

# %%
a, _ = preprocess(os.path.join('/kaggle/input/musicbpm','data', 'Testing', '164_Kuchizuke Diamond.mp3'), 164)

# %%
temp = a
plt.figure(figsize=(10,40))
for i, spectro in enumerate(temp[:len(temp) // 2]):
    plt.subplot(len(temp) // 2, 2, 1 + 2*i)
    plt.imshow(tf.transpose(spectro)[0])

for i, spectro in enumerate(temp[len(temp) // 2:]):
    plt.subplot(len(temp) // 2, 2, 2 + 2*i)
    plt.imshow(tf.transpose(spectro)[0])

# %%
def getDataPaths(*path):
    dataRelativePath = os.path.join(*path)

    supportedAudioFiles = ['/*.flac', '/*.wav', '/*.ogg', '/*.mp3', '/*.mp4a']
    supportedAudioFilesGlob = []

    for supportedAudioFile in supportedAudioFiles:
        supportedAudioFilesGlob.append(dataRelativePath + supportedAudioFile)

    dataPaths = tf.data.Dataset.list_files(supportedAudioFilesGlob, shuffle=False, seed=1234)
    return dataRelativePath, dataPaths

# %%
def getLabels(dataRelativePath, dataPaths):
    labels = []
    fileIndex = len(dataRelativePath)
    for elem in dataPaths:
        fileStringPath = elem.numpy().decode('utf-8')
        underscoreIndex = fileStringPath.find('_',fileIndex)
        labels.append(float(fileStringPath[fileIndex + 1:underscoreIndex]))
    
    return labels

# %%
def getData(*args):
    dataRelativePath, dataPaths = getDataPaths(*args)
    labels = getLabels(dataRelativePath, dataPaths)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((dataPaths, labels))

# %%
def datasetConfiguration(dataset, preprocessFunction):
    dataset = dataset.map(preprocessFunction)
    dataset = dataset.unbatch()
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=len(dataset) + 1)
    dataset = dataset.batch(16)
    dataset = dataset.prefetch(8)
    return dataset

# %%
def datasetPartition(dataset):
    trainingBatches = len(dataset) * 7 //10
    testBatches = len(dataset) - trainingBatches
    train = dataset.take(trainingBatches)
    test = dataset.skip(trainingBatches).take(testBatches)
    return train, test

# %%
data = getData('/kaggle/input/musicbpm','data', 'Training')

# %%
data = datasetConfiguration(data, preprocess)

# %%
train, test = datasetPartition(data)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, SeparableConv2D, InputLayer
from tensorflow.keras.layers import Dropout, BatchNormalization, MaxPool2D, GlobalMaxPool2D

# %%
def createModelTE(inputShape = (200, 100, 1)):
    model = Sequential(name="sequentialTE")
    model.add(InputLayer(inputShape))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(MaxPool2D((2,1)))
    
    model.add(SeparableConv2D(32, (3,3), activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())
    
    model.add(SeparableConv2D(64, (3,3), activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())
    
    model.add(SeparableConv2D(128, (3,3), activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())
    
    model.add(SeparableConv2D(256, (3,3), activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())
    
    model.add(GlobalMaxPool2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error', 
              metrics=[tf.keras.metrics.MeanAbsolutePercentageError(name="mape"), 
                       tf.keras.metrics.MeanSquaredError(name="mse")])
    return model
modelTe = createModelTE()
modelTe.summary()

# %%
def scheduler(epoch, lr):
    if lr <= 0.000001:
        return lr
    if epoch % 5 == 0 and epoch > 1:
        return lr * 0.8
    return lr

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=0.000001, verbose=1)
schedule_lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
early_s = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)

# %%
modelTe.fit(train, epochs=15, validation_data=test, callbacks=[reduce_lr, schedule_lr, early_s])

# %%
modelTe.save('/kaggle/working/musicFULLMODEL15E.h5')

# %%
modelTe.fit(train, epochs=15, validation_data=test, callbacks=[reduce_lr, schedule_lr, early_s])

# %%
modelTe.save('/kaggle/working/musicFULLMODEL30E.h5')

# %%
modelTe.fit(train, epochs=15, validation_data=test, callbacks=[reduce_lr, schedule_lr, early_s])

# %%
modelTe.save('/kaggle/working/musicFULLMODEL45E.h5')

# %%
modelTe.fit(train, epochs=15, validation_data=test, callbacks=[reduce_lr, schedule_lr, early_s])
modelTe.save('/kaggle/working/musicFULLMODEL60E.h5')

# %%
modelTe.fit(train, epochs=15, validation_data=test, callbacks=[reduce_lr, schedule_lr, early_s])
modelTe.save('/kaggle/working/musicFULLMODEL75E.h5')

# %%
modelTe.fit(train, epochs=15, validation_data=test, callbacks=[reduce_lr, schedule_lr, early_s])
modelTe.save('/kaggle/working/musicFULLMODEL90E.h5')

# %%
x = test.as_numpy_iterator().next()

# %%
au = modelTe.predict(x[0])

# %%
print(au)
x[1]

# %%
validationData = getData('/kaggle/input/musicbpm','data', 'Testing')
validationData = datasetConfiguration(validationData, preprocess)

# %%
xv = validationData.as_numpy_iterator().next()

# %%
yvh = modelTe.predict(xv[0])

# %%
print(yvh)
xv[1]


