# Face Recognition and Attendance System
This project aims to provide a solution for attendance management in classrooms by using facial recognition technology. The system detects and recognizes the faces of students in real-time and marks their attendance accordingly. It also captures the face moment of students and grades them based on how well they are paying attention in class.

Getting Started
To get started with this project, follow the steps below:

Clone the repository to your local machine
Install the required dependencies by running the following command in your terminal:

![image](https://user-images.githubusercontent.com/102272183/222970558-a95f8d5d-953d-4feb-884d-fb0c88790fe2.png)

Run the O_Dataset_MediaPipe.py file to create the dataset of student images
Run the O_Embeddings.py file to generate embeddings for each student in the dataset
Run the O_Grading.py file to start the face recognition and attendance system

## Dependencies
The following dependencies are required to run this project:

Python 3.6 or higher
TensorFlow
OpenCV
Mediapipe

## Files
O_Dataset_MediaPipe.py: This file is used to create the dataset of student images using MediaPipe
O_Embeddings.py: This file generates embeddings for each student in the dataset using the Facenet model
O_Grading.py: This file contains the code for face recognition and attendance management, as well as the grading system
License
This project is licensed under the MIT License - see the LICENSE.md file for details.

Acknowledgements
This project was inspired by this article on Analytics Vidhya.

## Contributing
Contributions to this project are welcome. If you find a bug or want to suggest an improvement, please open an issue or submit a pull request.
