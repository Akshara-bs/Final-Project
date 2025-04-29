# Final-Project
 **While viewing the colab Notebook directly on GitHub, it may encounter a rendering error stating "'state' key is missing from 'metadata.widgets'". It is because of the GitHub display limitation for interactive widgets. For proper viewing, Please Download the notebook.**

 Topic :
 **Disease Detection in Cassava Plant using Deep Learning-Based Image classification**

 This project focuses on the classification of cassava plant leaf disease using deep learning techniques. A hybrid model combining a custom CNN with MobileNetV2 was developed and compared with EfficientNetB0 and ResNet50 to assess performance, generalization, and computational efficiency.

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
•	A hybrid CNN-MobileNet architecture for efficiency.
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
The Hybrid Model achieved 64% accuracy, outperforming both EfficientNet (61%) and ResNet50 (61%) by capturing a broader range of features, improving classification accuracy.

**Methodology**
*Data Preprocessing*
    -Resized images to 128x128
   - Label encoding
   - Normalization
   - Train-validation-test split(training set-70%, Validation set-15%,Test set- 15%)
*Model architecture*
   - Hybrid Model with -Custom CNN block and MobileNetV2 as feature extractor -Fully connected classifier layer
   - EfficienetNetB0
   - ResNet50
*Training Configuration*
   -Epochs : 20
  -Batch size: 10
  -optimizer: Adam
 - Callbacks:ModelCheckpoint,ReduceLROnPlateau


**Evaluation Metrics**
    -Accuracy
    -Precision
    -Recall
    -F1-Score
     ROC Curve


**User Interface**
An interactive interface was developed using ipywidgets in google colab to:
- Upload cassava leaf images
- Display model predictions
- Visualize classification results

**Key Insights**
Hybrid CNN + MobileNetV2 demonstrated the best generalization in real-world conditions.
The model's low parameter count and fast training make it ideal for low-resource environments.
The interface allows practical field use for farmers and agronomists.
