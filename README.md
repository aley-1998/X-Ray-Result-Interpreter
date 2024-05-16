# X-Ray Result Interpreter

### Objectives:
In this project, a program was developed to classify chest X-rays. With a convolutional neural network-based classification model (CNN), lung infections in X-ray imaging are detected with great accuracy (87%). This model is able to classify scans into 1 of 2 categories: normal healthy lungs and infected unhealthy lungs.


### Methodologies/Techniques:
This project is implemented with Python programming language. The Python distribution Anaconda will is used since it contains all scientific Python libraries the program relies on to function, if not most. 
These libraries can be categorized as such: 
-	File handling (os, zipfile)
-	Data cleaning/handling (Pandas, NumPy)
-	Data visualization (Matplotlib, Seaborn)
-	Image enhancement (PIL, OpenCV)
-	Machine learning (TensorFlow, Keras, scikit-learn)
-	Performance monitoring and hyperparameter tuning (scikit-learn, Keras)
-	Model saving and deployment (pickle, Keras, Flask)
- Jupyter Lab will be my IDE of choice simply because I have used it the most in the past. However, IDE’s such as Spyder or Visual Studio are just as practical.
- The neural network will be created from scratch (not pre-trained). 
- Sensitivity analysis of the neural network will be performed to compare performance against other machine learning classification algorithms.
- GPU hardware accelaration will be used (either locally or through the cloud) for model training.
- Last, the model will be saved for deployment. 


### Dataset:
The model will train over a large dataset of X-ray images from anonymous patients (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). This dataset was created by a team of researchers from Qatar University, Qatar and University of Dhaka, Bangladesh for COVID-19 research. For each of the 4 categories, there are 2 folders of data (images and masks) as well as a metadata file with columns: filename, format, size and url. Sample screenshots of the dataset:

#### Metadata
![image](https://user-images.githubusercontent.com/69071476/229360061-179b120d-6357-471a-a06c-c9aa871066e7.png)

#### X-rays
![image](https://user-images.githubusercontent.com/69071476/229360000-0bc5a393-e3c6-4a03-879c-177426abe2c5.png)

#### Masks (In radiography, a mask is a version of the x-ray that does not include unwanted image details.)
![image](https://user-images.githubusercontent.com/69071476/229359414-58c34485-da95-42eb-9931-65a26d00d22b.png)

