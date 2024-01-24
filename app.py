# app.py
from PIL import Image
from keras_facenet import FaceNet
from numpy import asarray
from numpy import expand_dims
from yoloface import face_analysis
faceY = face_analysis()
import pickle
MyFaceNet = FaceNet()
sum=0

import mediapipe as mp
import streamlit as st
import numpy as np
import os
import cv2
from PIL import Image as Img
folder = 'training_data2/'
database = {}

MyFaceNet = FaceNet()

faceY = face_analysis()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def register_student(name,roll):
        # Initialize webcam
    cap = cv2.VideoCapture(0)
    progress_text = st.empty()
    startX, startY, endX, endY = 0, 0, 0, 0
    # Create a folder for the student's dataset
    dataset_path = f"training_data10/"
    assure_path_exists(dataset_path)

    count = 0
    while count < 2:  # Capture 50 frames (adjust as needed)
        ret, frame = cap.read()
        _, box, conf = faceY.face_detection(frame_arr=frame, frame_status=True, model='full')

        for i, rbox in enumerate(box):
            if conf[i] > 0.5:
                startX = rbox[0]
                startY = rbox[1]
                endX = rbox[0] + rbox[3]
                endY = rbox[1] + rbox[2]

        img_h, img_w, _ = frame.shape

        # Flip the image horizontally for a later selfie-view display
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Get the result
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                face_3d = []
                face_2d = []

                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if not frame.any():
                    st.error("Error: Empty frame. Unable to capture the image.")
                    break
                count += 1
                if y < -10:
                    cv2.imwrite(f"{dataset_path}{name}p.{roll}.{count}.jpg", frame[startY:endY, startX:endX])

                elif y > 10:
                    cv2.imwrite(f"{dataset_path}{name}p.{roll}.{count}.jpg", frame[startY:endY, startX:endX])

                else:
                    cv2.imwrite(f"{dataset_path}{name}f.{roll}.{count}.jpg", frame[startY:endY, startX:endX])

            # Update progress text
            progress_text.text(f"Images captured: {count}/2")

    cap.release()
    cv2.destroyAllWindows()

    # Display success message
    st.success("Dataset captured successfully!")

def generate_embeddings():

    folder = 'training_data10/'
    database = {}

    # MyFaceNet = FaceNet()

    for filename in os.listdir(folder):
        print("Embeddings Creating")
        path = folder + filename
        gbr1 = cv2.imread(folder + filename)

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

def grading():
    myfile = open("data.pkl", "rb")
    database = pickle.load(myfile)
    myfile.close()
    cap = cv2.VideoCapture(0)
    # Create Streamlit elements for dynamic updates
    grades_display = st.empty()
    student_display = st.empty()
    # initialize the dictionary to store grades for each identity
    # grades = {}
    Real_Grade = {}

    while (1):
        ret, frame = cap.read()

        _, box, conf = faceY.face_detection(frame_arr=frame, frame_status=True, model='full')
        for i, rbox in enumerate(box):
            if conf[i] > 0.5:
                startX = rbox[0]
                startY = rbox[1]
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
                    # cv2.imshow("cdcd", face)

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
                            Real_Grade[identity[:-1]] = 0.2
                    # Display real-time grades and student names

                    grades_display.write(f"Grades: {Real_Grade}")
                    student_display.write(f"Student: {identity[:-1]}")
                    # cv2.putText(frame, (identity[:-1] + " " + str("{:.2f}".format(Real_Grade[identity[:-1]]))), (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    # cv2.putText(frame, identity, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                except ValueError:
                    # Skip this iteration if there is an error extracting the face region from the frame
                    # print("null")
                    continue
        # cv2.imshow('res', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # print(Real_Grade)
        # print(grades)

    cv2.destroyAllWindows()
    cap.release()

def register():
    st.title("Student Registration")
    # Get student information
    name = st.text_input("Enter your name:")
    roll = st.text_input("Enter your roll number:")
    generate_embeddings_checkbox = st.checkbox("I agree to use my images for attendance detection and generate embeddings.")
    # print(generate_embeddings_checkbox)
    if st.button("Register"):
        # Display instructions
        st.write("Please look straight at the camera and tilt your head left and right for dataset capturing.")
        register_student(name,roll)
        # Display a checkbox to generate embeddings

        # print(generate_embeddings_checkbox)
        if generate_embeddings_checkbox:
            # Generate embeddings
            generate_embeddings()

            # Display success message
            st.success("Registration and embeddings generation completed successfully!")
    if st.button("Start Attendance"):
        grading()

# Streamlit App
if __name__ == "__main__":
    register()
