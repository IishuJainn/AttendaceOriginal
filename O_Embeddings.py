import os
import cv2
from PIL import Image as Img
# from numpy import asarray
import numpy as np
from keras_facenet import FaceNet
import pickle
folder = 'training_data2/'
database = {}

MyFaceNet = FaceNet()

for filename in os.listdir(folder):

    path = folder + filename
    gbr1 = cv2. imread(folder + filename)

    gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
    gbr = Img.fromarray(gbr)
    gbr_array = np.asarray(gbr)
    face = Img.fromarray(gbr_array)
    face = face.resize((160, 160))
    face = np.asarray(face)

    face = np.expand_dims(face, axis=0)
    signature = MyFaceNet.embeddings(face)


    database[os.path.splitext(filename)[0].split(".")[0]] = signature
    # database[os.path.splitext(filename)[0]] = signature

myfile = open("data.pkl", "wb")
pickle.dump(database, myfile)
myfile.close()
