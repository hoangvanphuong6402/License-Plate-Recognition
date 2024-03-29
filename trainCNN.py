import cv2
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

digit_w = 30
digit_h = 60

write_path = "License-Plate-Recognition/data/"


def get_digit_data(path):
    digit_list = []
    label_list = []

    for number in range(10):
        i = 0
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            img = img.reshape(digit_h, digit_w, 1)

            digit_list.append(img)
            label_list.append(int(number))

    for number in range(65, 91):
        print(number)
        i = 0
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            img = img.reshape(digit_h, digit_w, 1)

            digit_list.append(img)
            label_list.append(int(number))

    return digit_list, label_list


# lấy dữ liệu
digit_path = "License-Plate-Recognition/data/"
digit_list, label_list = get_digit_data(digit_path)

digit_list = np.array(digit_list, dtype=np.float32)
label_list = np.array(label_list)

# One-hot encode the target column
label_list = to_categorical(label_list)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(digit_list, label_list, test_size=0.15)

# Create model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(digit_h, digit_w, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(91, activation='softmax'))

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
H = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50)

# Save the model
model.save('License-Plate-Recognition\cnn_model.h5')

# Vẽ đồ thị loss, accuracy của training set và validation set
fig = plt.figure()
numOfEpoch = 50
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()

plt.show()

