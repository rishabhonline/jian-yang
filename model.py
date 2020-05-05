import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.applications.resnet import preprocess_input
# from tensorflow.python.keras.applications import ResNet50
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Dropout, GlobalAveragePooling2D
import itertools

# print(help(tf.python.keras.applications))



train_path = 'data/seefood/train'
test_path = 'data/seefood/test'

hot_dog_path = 'data/seefood/train/hot_dog'
not_hot_dog_path = 'data/seefood/train/not_hot_dog'

train_data_hd = [os.path.join(hot_dog_path, filename)
              for filename in os.listdir(hot_dog_path)]
train_data_nhd = [os.path.join(not_hot_dog_path, filename)
              for filename in os.listdir(not_hot_dog_path)]

img_size = 224
num_classes = 2

data_generator = ImageDataGenerator()

train_generator = data_generator.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=498,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=500,
        class_mode='categorical')

def read_and_prep_images(img_paths, img_height=img_size, img_width=img_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)

train_data = read_and_prep_images(train_data_nhd)
train_data1 = read_and_prep_images(train_data_hd)


data = train_generator[0][0][3]
plt.imshow(data, interpolation='nearest')
plt.show()

model = Sequential()
model.add(Conv2D(8, kernel_size=(3,3),
#             strides=2,
            activation='relu',
            input_shape=(img_size, img_size, 3)))
Dropout(.5)
model.add(Conv2D(8, kernel_size=(3,3),
#                 strides=2,
                activation='relu'))
Dropout(.5)

model.add(Conv2D(8, kernel_size=(3,3),
#                 strides=2,
                activation='relu'))
model.add(Conv2D(8, kernel_size=(3,3),
#                 strides=2,
                activation='relu'))
model.add(Conv2D(8, kernel_size=(3,3),
#                 strides=2,
                activation='relu'))
model.add(Conv2D(8, kernel_size=(3,3),
#                 strides=2,
                activation='relu'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer='adam',
             metrics=['accuracy'])

input_x = (train_generator[0][0]/255)
input_y = (train_generator[0][1])

model.fit(input_x,
         input_y,
         batch_size=24,
         epochs=5)

# model.fit_generator(train_generator,
#         steps_per_epoch=5,
#         validation_data=validation_generator,
#         validation_steps=1)

output_x = (validation_generator[0][0]/255)
output_y = validation_generator[0][1]

print(model.evaluate(output_x, output_y))

pre1 = model.predict(train_data1)
print("Hot Dog")
for i in range(10):
    #display(Image(train_data_hd[i]))
#     print(pre1[i][0], pre1[i][1])
    print(np.argmax(pre1[i]))

pre = model.predict(train_data)
print("Not Hot Dog")
for i in range(10):
    #display(Image(train_data_nhd[i]))
    print(np.argmax(pre[i]))

model.save('saved_model/my_model') 