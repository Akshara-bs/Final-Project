# Final-Project
♦️ **While viewing the colab Notebook directly on GitHub, it may encounter a rendering error stating "unable to rendor code file". It is because of the GitHub display limitation for interactive widgets. For proper viewing, Please Download the notebook.**♦️

 Topic :
 **Disease Detection in Cassava Plant using Deep Learning-Based Image classification**

 This project focuses on the classification of cassava plant leaf disease using deep learning techniques. A  custom CNN was developed and compared with EfficientNetB0 and ResNet50 to assess performance, generalization, and computational efficiency.

The research question guiding the initiative is therefore:
*How can machine learning techniques be applied to accurately classify images of cassava plants into specific disease categories, and what is the most effective model architecture for this task?* 

**Aim:** 
To develop an accurate and accessible deep learning-based system for automatic classification of cassava plant diseases from images.    

**Key Objectives:**
•	preprocessing and analyzing the dataset
•	applying  data augmentation techniques, 
•	Design and evaluate multiple deep learning architectures, including:
•	A custom Convolutional Neural Network (CNN)
•	Transfer learning models (EfficientNet, ResNet)
•	Assess performance using metrics like precision, recall, F1-score, and accuracy.
•	Develop a user-friendly interface for real-time image uploads and predictions.

 **Dataset**
 The project’s dataset in Crop ‘Diseases Classification Dataset’, was obtained from kaggle (https://www.kaggle.com/datasets/mexwell/crop-diseases-classification/data), and is derived from the Cassava Leaf Disease Dataset(https://www.kaggle.com/c/cassava-leaf-disease-classification).
 It contains high-quality images of cassava leaves classified into five categories: 
  Cassava Bacterial Blight (CBB)
  - Cassava Brown Streak Disease (CBSD)
  - Cassava Green Mottle (CGM)
  - Cassava Mosaic Disease (CMD)
  - Healthy
The photographs are categorised and structured using two main elements: a train.csv file which associates image filenames with their corresponding illness diagnoses, and a train_images/ directory that has the real leaf images.

**Models Used**
The Custom CNN Model achieved 76% accuracy, and other pretrained model such as efficientnet and resnet occur accuracies 82% and 78% respectively by capturing a broader range of features, improving classification accuracy.

**Methodology**
*Data Preprocessing*
-Resized images to 150x150 .
 -Label encoding for categorical labels.
 -Normalization of pixel values (scaling to 0-1 for the Custom CNN, and using ImageNet mean/std for EfficientNetand resnet).
 -Train-validation split (80% training, 20% validation) for all models
- Oversampling was applied to the training set of the Custom CNN to balance class distribution. Weighted Random Sampler was used for the training set of the EfficientNet and resnet models to handle class imbalance during training.
-Data Augmentation was applied to the training data using ImageDataGenerator for the Custom CNN and torchvision.transforms for EfficientNet and resnet.

*Model architecture*
   -Custom CNN
Sequential model with convolutional, max pooling, batch normalization, and dropout layers.
Flatten layer followed by dense layers for classification.
-EfficientNetB0 and resnet
Pre-trained EfficientNetB0 model loaded from torchvision.models.
Modified the classifier layer to have 5 output classes.




**Evaluation Metrics**
    -Accuracy
    -Precision
    -Recall
    -F1-Score
   


**User Interface**
An interactive interface was developed using ipywidgets in google colab to:
- Upload cassava leaf images
- Display model predictions
- Visualize classification results

**Key Insights**
The code trains and evaluates two different models: a Custom CNN and a pre-trained EfficientNetB0 and resnet.
Data balancing techniques (oversampling for CNN, Weighted Random Sampler for EfficientNet and resnet) were used to address class imbalance.
The code includes visualizations of the data distribution before and after balancing (simulated for EfficientNet and resnet).
Training history (loss and accuracy) is plotted for all models.
Detailed evaluation metrics (Classification Report and Confusion Matrix) are provided for the validation set for all models.
A basic command-line style interface within the notebook allows for prediction on individual images using the trained model.
