#%%
import os
import  re
import tensorflow as tf
import tensorflow_io as tfio

with tf.device("/device:CPU:0"):
    # lista koja ce imati sve labele glasova
    labels = list();

    # path do foldera labela
    folderPath = "../Data/lab/"
    # vraca listu imena svih datoteka u folderu
    filesLIst = os.listdir(folderPath)

    # Otvara svaki file
    for file in filesLIst:
        txtFilePath = os.path.join(folderPath, file)

        # ucitavanje linija
        try:
            with open(txtFilePath, 'r') as file:
                # Perform operations on the file
                lines = file.readlines()

        except FileNotFoundError:
            print("File not found.")

        # Iterira kroz retke filea
        for line in lines:
            # Mice \n charactere
            line = line[:-1]
            # Regex da se uzme sve poslje drugog razmaka
            result = re.search(r'^(?:\S+\s){2}(.*)$', line)
            newLabel = result.group(1)

            # ako ne postoji u listi dodaj u listu
            if newLabel not in labels:
                labels.append(newLabel)
    labels.sort()
    print(labels)

    # napravi rijecnik
    labelsDict = {label: index for index, label in enumerate(labels)}
    print(labelsDict)
#%%
with tf.device("/device:CPU:0"):
    # lista koja ce imati tuple-ove (audioSample, indexLabele)
    data = list()

    labFolderPath = "../Data/lab/"
    wavFolderPath = "../Data/wav/"


    # lista lab i wav fileova
    labFiles = os.listdir(labFolderPath)
    wavFiles = os.listdir(wavFolderPath)

    #sample rate, pracnje progessa, broj errora
    rate = 0
    i = 0
    er = 0

    # obrisi fajlove iz liste koji nisu spareni
    labAnfWavFilesFixed = list()
    # prolazi kroz svaki lab file  za svaki wav file te ako postoji match appenda u listu
    for wavFile in wavFiles:
        match = re.search('(.*)\.', wavFile)
        if match:
            wavText = match.group(1)

        for labFile in labFiles:
            match = re.search('(.*)\.', labFile)
            if match:
                labText = match.group(1)
            if labText == wavText:
                labAnfWavFilesFixed.append((wavFile, labFile))

    print(labAnfWavFilesFixed)


    # iterira kroz wav i lab fileove
    for wavFile, labFile in labAnfWavFilesFixed:
        try:
            filename = os.path.join(wavFolderPath, wavFile)
            # otvori audio file, ako baci error preskoci
            print(wavFile + " " + labFile)

            audioFile = tfio.audio.AudioIOTensor(filename)
            rate = audioFile.rate


            # otvori lab file iteracija kroz linije
            with open(os.path.join(labFolderPath, labFile), 'r') as file:
                lines = file.readlines()
                print(wavFile + " " + labFile)
                for line in lines:
                    # regex za pocetak i kraj sample-a
                    match = re.match(r'(\d+)\s+(\d+)', line)
                    if match:
                        splitStart = float(match.group(1))
                        splitEnd = float(match.group(2))

                    line = line[:-1]
                    # Regex da se uzme sve poslje drugog razmaka, labela
                    result = re.search(r'^(?:\S+\s){2}(.*)$', line)
                    label = result.group(1)

                    splitStart = splitStart / 10000000 * 16000
                    splitEnd = splitEnd / 10000000 * 16000

                    audioSliced = audioFile[int(splitStart):int(splitEnd)]

                    audio_tensor = tf.squeeze(audioSliced, axis=[-1])

                    sampleLength = tf.size(audio_tensor).numpy()

                    # maksimalna duzinu samplea!
                    if sampleLength < 3000 and label not in ['uzdah', 'sil', 'greska', 'buka']:
                        data.append([audio_tensor, label])
            i += 1
            print(i, len(wavFiles))
            """if i > 700:
                break"""

        # ako se dogodi greska skipaj na sljedecu iteraciju
        except Exception as  e:
            er = er + 1
            print("error")
            print(e)
            continue
#%%
import numpy as np
import copy
with tf.device("/device:CPU:0"):
    # Trazi maksimalnu duljinu sample-a
    max = 0
    maxValues = list()
    for i, sample in enumerate(data):
        sampleSize = tf.size(sample[0]).numpy()
        if sampleSize > max:
            max = sampleSize
            maxValues.append(sampleSize)
            print(i, "i")



    # Zero padding
    for i, sample in enumerate(copy.deepcopy(data)):
        sampleSize = tf.size(sample[0]).numpy()
        # Create tensor with zeros
        zerosTensor = tf.constant(np.zeros(max - sampleSize, dtype=int), dtype=tf.int16)
        newTensor = tf.concat([sample[0], zerosTensor], axis=0)

        data[i][0] = newTensor


    print(maxValues)

#%%
import matplotlib.pyplot as plt
from IPython.lib.display import Audio
import numpy as np

def plot_waveform(audio_tensor, sample_rate):
    audio_np = audio_tensor.numpy()
    time_samples = np.arange(audio_np.shape[0]) / sample_rate

    plt.figure(figsize=(10, 4))
    plt.plot(time_samples, audio_np)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.grid(True)
    plt.show()

audio_np = 1
i = 0
for index, sample in enumerate(data):
    if i == 10:
        break
    if sample[1] == 'b':
        plot_waveform(sample[0], rate)
        audio_np = sample[0].numpy()
        if i == 5:
            audio_np2 = audio_np
        Audio(data=audio_np, rate=16000)
        i += 1
#%%
Audio(data=audio_np, rate=16000)
#%%
Audio(data=audio_np2, rate=16000)
#%%
"""import sys
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())     ## this command list all the processing device GPU and CPU


device_name = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
if device_name[0] == "/device:GPU:0":
    device_name = "/gpu:0"
    print('GPU')
else:
    print('CPU')
    device_name = "/cpu:0"
"""
#%%
# create spectograms
with tf.device("/device:CPU:0"):
    def get_spectrogram(waveform):
        #STFT.
        spectrogram = tf.signal.stft(
            waveform, frame_length=128, frame_step=64)

        spectrogram = tf.abs(spectrogram)
        # Dodavnaje jos jedne dimenzije
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram


    spectData = list()
    for index, sample in enumerate(data):
        print(index)
        spectrogram = get_spectrogram(sample[0].numpy().astype(float))
        spectData.append([spectrogram, sample[1]])


    #del data
#%%
from IPython.display import Audio, display

i = 0
for dataSample, spectDataSample in zip(data, spectData):
    print("LABELE: " + str(dataSample[1]) + " " + str(spectDataSample[1]))
    print("WAVEFORM SHAPE: " + str(dataSample[0].shape))
    print("SPECT SHAPE: " + str(spectDataSample[0].shape))


    if i == 20000:
        break
    i += 1
#%%
import matplotlib.pyplot as plt
def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

i = 0
for index, sample in enumerate(spectData):
    if sample[1] == 'b':
        fig, ax = plt.subplots()
        plot_spectrogram(sample[0].numpy(), ax)
        plt.xlabel("Vremenska os")
        plt.ylabel("Frekvencijska os")
        plt.show()
        i += 1
    if i == 10:
        break
#%%
with tf.device("/device:CPU:0"):
    # napravi nove liste za spectograme i labele
    sampleLabels = list()
    sampleSpect = list()
    for sample in spectData:
        sampleLabels.append(sample[1])
        sampleSpect.append(sample[0])


    indexLabels = list()
    for sampleLabel in sampleLabels:
        indexLabels.append(labelsDict[sampleLabel])


    #del spectData


#%%
print(type(sampleSpect[1]))
print(type(indexLabels[1]))
#%%
# create tfDataset
with tf.device("/device:CPU:0"):
    tfDataset = tf.data.Dataset.from_tensor_slices((sampleSpect, indexLabels))
    print(tfDataset.element_spec)

    #del sampleSpect, sampleLabels


#%%
for i, element in enumerate(tfDataset):
    if i == 50000:
        break
    print(element[1])

print(np.shape([
    [1, 2, 3],
    [1, 2, 3]
]))
#%%
with tf.device("/device:CPU:0"):
    # Shuffle the dataset
    dataset_shuffled  = tfDataset.shuffle(buffer_size=1000, seed=42)

    # Izracun duljina train, val i test datasetova
    total_size = len(tfDataset)
    train_size = int(0.7 * total_size)  # 70% for training
    val_size = int(0.2 * total_size)  # 20% for validation
    test_size = total_size - train_size - val_size  # Remaining 10% for testing

    # raspodjela
    train_dataset = dataset_shuffled.take(train_size)
    test_and_val_dataset = dataset_shuffled.skip(train_size)
    val_dataset = test_and_val_dataset.take(val_size)
    test_dataset = test_and_val_dataset.skip(val_size)

    print(len(list(train_dataset)), len(list(val_dataset)), len(list(test_dataset)))

#%%
# izrada batcheva
with tf.device("/device:CPU:0"):
    train_dataset = train_dataset.batch(8)
    val_dataset = val_dataset.batch(8)
    test_dataset = test_dataset.batch(8)
#%%
print(train_dataset.element_spec)
#%%
input_shape = train_dataset.element_spec[0].shape
print('Input shape:', input_shape[:])

for i, element in enumerate(train_dataset):
    print(element)
    if i == 2:
        break
#%%
# Izrada modela
with tf.device("/device:GPU:0"):
    from tensorflow.keras import layers
    from tensorflow.keras import models

    input_shape = train_dataset.element_spec[0].shape[1:]
    print('Input shape:', input_shape)
    num_labels = len(labels)

    # normalizacija
    norm_layer = layers.Normalization()
    norm_layer.adapt(data=train_dataset.map(map_func=lambda spec, label: spec))
    """
    # ORG MODEL
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])"""


    # VECA KOMPLEKSNOST
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_labels)
    ])

    """
    # VELIKA KOMPLEKSNOST
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Conv2D(512, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])
    """

    model.summary()

#%%
with tf.device("/device:GPU:0"):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

#%%
with tf.device("/device:GPU:0"):
    EPOCHS = 10
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
    )
#%%

#Palatali
palataliSpect = list()
palataliLabels = list()

#Samoglasnici
samoglasniciSpect = list()
samoglasniciLabels = list()

# Slovo Č
CSpect = list()
CLabels = list()

#Slovo a, 6
aSpect = list()
aLabels = list()

#Slovo c, 10
cSpect = list()
cLabels = list()

#Slovo b, 8
bSpect = list()
bLabels = list()

#Slovo h, 19
hSpect = list()
hLabels = list()

for spect, label in test_dataset:
    for i in range(len(spect)):
        # Palatali
        if label[i].numpy() in (13, 11, 5, 4, 3, 2, 1, 0, 22):
            print(type(spect[i]))
            print(type(int(label[i])))
            palataliSpect.append(spect[i])
            palataliLabels.append(int(label[i]))

        #Samoglasnici
        if label[i].numpy() in (36, 35, 28, 27, 21, 20, 15, 14, 7, 6):
            print(type(spect[i]))
            print(type(int(label[i])))
            samoglasniciSpect.append(spect[i])
            samoglasniciLabels.append(int(label[i]))

        if label[i].numpy() == 8:
            bSpect.append(spect[i])
            bLabels.append(int(label[i]))

        if label[i].numpy() == 19:
            hSpect.append(spect[i])
            hLabels.append(int(label[i]))

        if label[i].numpy() == 0:
            CSpect.append(spect[i])
            CLabels.append(int(label[i]))

        if label[i].numpy() == 6:
            aSpect.append(spect[i])
            aLabels.append(int(label[i]))

        if label[i].numpy() == 10:
            cSpect.append(spect[i])
            cLabels.append(int(label[i]))

palataliDataset = tf.data.Dataset.from_tensor_slices((palataliSpect, palataliLabels))
samoglasniciDataset = tf.data.Dataset.from_tensor_slices((samoglasniciSpect, samoglasniciLabels))
CDataset = tf.data.Dataset.from_tensor_slices((CSpect, CLabels))
aDataset = tf.data.Dataset.from_tensor_slices((aSpect, aLabels))
cDataset = tf.data.Dataset.from_tensor_slices((cSpect, cLabels))
bDataset = tf.data.Dataset.from_tensor_slices((bSpect, bLabels))
hDataset = tf.data.Dataset.from_tensor_slices((hSpect, hLabels))
#%%
print(len(list(palataliDataset)))
palataliDataset = palataliDataset.batch(8)
samoglasniciDataset = samoglasniciDataset.batch(8)
CDataset = CDataset.batch(8)
aDataset = aDataset.batch(8)
cDataset = cDataset.batch(8)
bDataset = bDataset.batch(8)
hDataset = hDataset.batch(8)

#%%
palataliAcc = model.evaluate(palataliDataset)[1]
samoglasniciAcc = model.evaluate(samoglasniciDataset)[1]
CAcc = model.evaluate(CDataset)[1]
aAcc = model.evaluate(aDataset)[1]
cAcc = model.evaluate(cDataset)[1]
bAcc = model.evaluate(bDataset)[1]
hAcc = model.evaluate(hDataset)[1]
#%%
percentages = [palataliAcc, samoglasniciAcc, CAcc, aAcc, cAcc, bAcc, hAcc]
labels = ["palatali", "samoglasnici", "Č", "a", "c", "b", "h"]

percentages = [round(p*100) for p in percentages]


plt.bar(labels, percentages)


for i in range(len(percentages)):
    plt.text(i, percentages[i] + 0.5, str(percentages[i])+'%', ha='center')

plt.xlabel('Labele')
plt.ylabel('Postotak točnih predikcija (%)')
plt.title('Točnost predikcija za određene glasove ili grupe glasova')
plt.ylim(0, 100)
plt.xticks(rotation=90)

plt.show()
#%%
model.evaluate(test_dataset)