import cv2
import mediapipe as mp
import numpy as np
import time
import os
from yoloface import face_analysis

face = face_analysis()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

name = input("Enter your name: ")
Roll = input("Enter your roll no: ")


count = 0

assure_path_exists("training_data10/")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    _, box, conf = face.face_detection(frame_arr=frame, frame_status=True, model='full')
    for i, rbox in enumerate(box):
        if conf[i] > 0.5:
            startX = rbox[0]
            startY = rbox[1]
            endX = rbox[0] + rbox[3]
            endY = rbox[1] + rbox[2]
            # output_frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        parent_dir = "training_data10"
        child_dir = str(name)
        if not os.path.exists(parent_dir):
            os.makedirs(path)

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
            face_3d = []
            face_2d = []

            for idx, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])

            # Convert to NumPy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            count += 1
            if y < -10:
                cv2.imwrite("training_data10/" + str(name) + 'p' + '.' + str(Roll) + '.' + str(count) + ".jpg",frame[startY:endY, startX:endX])

            elif y > 10:
                cv2.imwrite("training_data10/" + str(name) + 'p' + '.' + str(Roll) + '.' + str(count) + ".jpg",frame[startY:endY, startX:endX])

            else:
                cv2.imwrite("training_data10/" + str(name) + 'f' + '.' + str(Roll) + '.' + str(count) + ".jpg",frame[startY:endY, startX:endX])


    # Display the image
    cv2.imshow('Face Mesh', frame)



    # Exit the program
    if cv2.waitKey(5) & 0xFF == 27:
        break
    elif count > 600:
        break
cap.release()
cv2.destroyAllWindows()

