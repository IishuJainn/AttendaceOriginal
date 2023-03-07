from PIL import Image
from keras_facenet import FaceNet
from keras.models import load_model
import numpy as np
from numpy import asarray
from numpy import expand_dims

from yoloface import face_analysis

faceY = face_analysis()

import pickle
import cv2

MyFaceNet = FaceNet()
sum=0

myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()

cap = cv2.VideoCapture(0)

# initialize the dictionary to store grades for each identity
# grades = {}
Real_Grade={}

while (1):
    ret, frame = cap.read()

    _, box, conf = faceY.face_detection(frame_arr=frame, frame_status=True, model='full')
    for i,rbox in enumerate(box):
        if conf[i] > 0.5:
            startX=rbox[0]
            startY=rbox[1]
            endX = rbox[0] + rbox[3]
            endY = rbox[1] + rbox[2]
            output_frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            try:
                gbr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gbr = Image.fromarray(gbr)
                gbr_array = asarray(gbr)

                face = gbr_array[startY:endY, startX:endX]

                face = Image.fromarray(face)
                face = face.resize((160, 160))
                face = asarray(face)
                cv2.imshow("cdcd",face)

                face = expand_dims(face, axis=0)
                signature = MyFaceNet.embeddings(face)

                min_dist = 100
                identity = ' '
                for key, value in database.items():
                    dist = np.linalg.norm(value - signature)

                    if dist < min_dist:
                        min_dist = dist
                        identity = key

                # update the grade for the current identity
                if identity[-1] == 'f':
                    if identity[:-1] in Real_Grade:
                        # grades[identity] += 0.2
                        Real_Grade[identity[:-1]] += 0.2
                    else:
                        # grades[identity] = 0.2
                        Real_Grade[identity[:-1]] = 0.2

                else:
                    if identity[:-1] in Real_Grade:
                        # grades[identity] -= 0.2  # decrease grade if identity is already seen
                        Real_Grade[identity[:-1]] -= 0.2
                    else:
                        # grades[identity] = 0.0
                        Real_Grade[identity[:-1]]  = 0.2

                # cv2.putText(frame, (identity[:-1] + " " + str("{:.2f}".format(Real_Grade[identity[:-1]]))), (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                # cv2.putText(frame, identity, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            except ValueError:
                # Skip this iteration if there is an error extracting the face region from the frame
                print("null")
                continue
    cv2.imshow('res', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    print(Real_Grade)
    # print(grades)

cv2.destroyAllWindows()
cap.release()