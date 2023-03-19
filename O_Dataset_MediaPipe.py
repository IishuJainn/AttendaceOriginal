import cv2
import mediapipe as mp
import numpy as np
import time
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

name = input("Enter your name: ")
Roll = input("Enter your roll no: ")


count = 0

assure_path_exists("training_data2/")

# Load the pre-trained SSD model
model = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()

    # Rescale the frame for processing
    resized_frame = cv2.resize(frame, (300, 300))

    # Construct a blob from the frame
    blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network to get the detections
    model.setInput(blob)
    detections = model.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence score and the bounding box coordinates
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(startX, 0)
            startY = max(startY, 0)
            endX = min(endX, frame.shape[1] - 1)
            endY = min(endY, frame.shape[0] - 1)


            # Draw the bounding box and the confidence score on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)



        parent_dir = "training_data2"
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
                cv2.imwrite("training_data2/" + str(name) + 'p' + '.' + str(Roll) + '.' + str(count) + ".jpg",frame[startY:endY, startX:endX])

            elif y > 10:
                cv2.imwrite("training_data2/" + str(name) + 'p' + '.' + str(Roll) + '.' + str(count) + ".jpg",frame[startY:endY, startX:endX])

            else:
                cv2.imwrite("training_data2/" + str(name) + 'f' + '.' + str(Roll) + '.' + str(count) + ".jpg",frame[startY:endY, startX:endX])


    # Display the image
    cv2.imshow('Face Mesh', frame)



    # Exit the program
    if cv2.waitKey(5) & 0xFF == 27:
        break
    elif count > 300:
        break
cap.release()
cv2.destroyAllWindows()

