#!/usr/bin/env python
# coding: utf-8

# In[1]:


import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import keras
from keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import json as json
from flask_cors import CORS, cross_origin

# In[ ]:


app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

def get_model():
    global model
    model = load_model('covid_testing.h5')
    print(" * Model loaded!")


# In[ ]:

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

print("* Loading Keras model.....")
get_model()

@app.route("/predict", methods=["POST"])
# @cross_origin()
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224,224))

    prediction = model.predict(processed_image).tolist()
    prediction_class = "yes" if prediction[0][0] else "no"
    # print(prediction)
    response = {
        'prediction': {
            'Covid': prediction_class,
            # 'Normal': "no"
        }
    }
    return jsonify(response)




# In[ ]:





# In[ ]:





# In[ ]:




