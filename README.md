# X-Ray-Result-Interpreter

### Introduction:
Often, doctors and radiologists properly diagnose their patients according to their X-ray scans. But sometimes patients are misdiagnosed because of human error: either the radiologist makes an erroneous interpretation of the X-rays or the scan itself was not done properly leading to false interpretations. Other times, radiologists ask their patients to re-do the scans because they do not look clear enough to interpret, which is time consuming for both patients and doctors as appointments get pushed back and rescheduled. Patients must reschedule more scan appointments and pay more money. Sick patients do not get their diagnosis in a timely manner. In extreme cases, a quick and accurate diagnosis can save a patient’s life.


### Objectives:
The aim of this project is to develop a program which eliminates uncertainty for radiologists when interpreting unclear chest X-rays. I will implement a convolutional neural network-based classification model that detects lung infections in X-ray imaging with high accuracy. This model will be able to classify scans into 1 of 2 categories: a normal healthy lung and an infected unhealthy lung.


### Methodologies/Techniques:
This project will be implemented with Python programming language, as it is a powerful high level scientific programing language. The Python distribution Anaconda will be used since it contains all core Python libraries the program relies on to function, if not most. 
These libraries can be categorized as such: 
-	File handling (os, zipfile)
-	Data cleaning/handling (Pandas, NumPy)
-	Data visualization (Matplotlib, Seaborn)
-	Image enhancement (PIL, OpenCV)
-	Machine learning (TensorFlow, Keras, scikit-learn)
-	Performance monitoring and hyperparameter tuning (scikit-learn, Keras)
-	Model saving and deployment (pickle, Keras, Flask)
Jupyter Lab will be my IDE of choice simply because I have used it the most in the past. However, IDE’s such as Spyder or Visual Studio are just as practical.
The neural network will be created from scratch (not pre-trained). 
Sensitivity analysis of the neural network will be performed to compare performance against other machine learning classification algorithms, part of the performance monitoring procedure.
If time allows, I will try to utilize a local GPU to accelerate model training. If that will not be possible, I will use a cloud service such as Google Colab for training instead.
After training, the model will be saved for deployment. 


### Dataset:
The model will train over a large dataset of X-ray images from anonymous patients (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). This dataset was created by a team of researchers from Qatar University, Qatar and University of Dhaka, Bangladesh for COVID-19 research. For each of the 4 categories, there are 2 folders of data (images and masks) as well as a metadata file with columns: filename, format, size and url. Sample screenshots of the dataset:

![image](https://user-images.githubusercontent.com/69071476/229360061-179b120d-6357-471a-a06c-c9aa871066e7.png) ----> Metadata


![image](https://user-images.githubusercontent.com/69071476/229360000-0bc5a393-e3c6-4a03-879c-177426abe2c5.png) ----> X-rays

![image](https://user-images.githubusercontent.com/69071476/229359414-58c34485-da95-42eb-9931-65a26d00d22b.png) ----> Masks (In radiography, a mask is a version of the x-ray which does not include unwanted areas.)

