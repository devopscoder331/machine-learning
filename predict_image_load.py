import keras
from keras.models import load_model
import numpy as np
import tensorflow as tf

def dog_cat_predict(model, image_file):
    label_names = ["cat", "dog", "panda"]
    img = keras.preprocessing.image.load_img(image_file, target_size=(150, 150))
    img_arr = np.expand_dims(img, axis=0) / 255.0
    result = model.predict_classes(img_arr)
    print(result)
    print("Result: %s" % label_names[result[0]])

model = load_model('cnn_small_categorical.h5')
print(model.summary())
model = tf.keras.models.load_model('cnn_small_categorical.h5')
dog_cat_predict(model, "image")
