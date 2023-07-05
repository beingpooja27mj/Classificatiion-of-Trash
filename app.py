import numpy as np
import os
import keras.utils as image
import tensorflow as tf
from keras.models import load_model
from flask import Flask,render_template,request
app=Flask(__name__)

model=load_model("garbage.h5")
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img= tf.keras.utils.load_img(filepath,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        index=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        text="The predicted garbage is : " +str(index[pred[0]])
    return text

if __name__=='__main__':
    app.run()