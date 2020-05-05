import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.resnet import preprocess_input
import numpy as np
import scipy

st.markdown("""
	# Greatest Coder Living 🧑🏻‍💻 @ Jian Yang
![enter image description here](https://pbs.twimg.com/media/C_4cr03XoAA33Xp?format=jpg&name=small)
#  Jian Yang: Hotdog Identifying App 🌭
Jin Yang, "What would you say if I told you there is an app on the market .." when he gets interrupted by Erlich.
Lucky that I have no Erlich to interrupt me.
# Why This App?
While it might seem trivial, the reason why I decided to do this mini project which took me less than one night to complete is that often, Data Scientists and Machine Learning Engineers have a very good performing model. However, due to lack of presenattion skills, the project ends up getting boring. Unless you have a good web developer things might get a little tricky if jsut AI engineers were to sit and code front end web interfaces which obviously is not their job.

[Streamlit’s](https://www.streamlit.io/) open-source app framework  is  the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours! All in pure Python. All for free. Isn't open-soure the best! 

	""")


def read_and_prep_images(img, img_height=224, img_width=224):
    imgs = load_img(img, target_size=(img_height, img_width))
    img_array = np.array([img_to_array(imgs)])
    output = preprocess_input(img_array)
    return(output)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
	model = tf.keras.models.load_model('saved_model/my_model')
	image = Image.open(uploaded_file)
	st.image(image, caption='Uploaded Image.', use_column_width=True)
	image = image.resize((224,224), Image.ANTIALIAS)
	st.image(image, caption='Uploaded Image.', use_column_width=True)
	pred = model.predict(preprocess_input(np.array([img_to_array(image)])))
	st.write(pred)
	if(pred[0,0]>pred[0,1]):
		st.write('Hot Dog 🌭')	
	else:
		st.write('Not a Hot Dog ⛔ 🌭 ')


