import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.resnet import preprocess_input
import numpy as np

def read_and_prep_images(img, img_height=224, img_width=224):
    imgs = load_img(img, target_size=(img_height, img_width))
    img_array = np.array([img_to_array(imgs)])
    output = preprocess_input(img_array)
    return(output)

model = tf.keras.models.load_model('saved_model/my_model')
paths = ['data/test/New Folder/133012.jpg', 'data/test/New Folder/burger.jpg']
for p in paths:
	pre1 = model.predict(read_and_prep_images(p, img_height=224, img_width=224))
	print(pre1)
# Check its architecture
# model.summary()

