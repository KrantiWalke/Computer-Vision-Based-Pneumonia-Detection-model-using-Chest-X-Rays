# Computer-Vision-Projects

# Pneumonia Detection Challenge
## The Real Problem
### What is Pneumonia?
Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it. The infection causes inflammation in the air sacs in your lungs, which are called alveoli.Pneumonia accounts for over 15% of all deaths of children under 5 years old internationally. In 2017, 920,000 children under the age of 5 died from the disease. It requires review of a chest radiograph (CXR) by highly trained specialists and confirmation through clinical history, vital signs and laboratory exams. Pneumonia usually manifests as an area or areas of increased opacity on CXR. However, the diagnosis of pneumonia on CXR is complicated because of a number of other conditions in the lungs such as fluid overload (pulmonary edema), bleeding, volume loss (atelectasis or collapse), lung cancer, or post-radiation or surgical changes. Outside of the lungs, fluid in the pleural space (pleural effusion) also appears as increased opacity on CXR. When available, comparison of CXRs of the patient taken at different time points and correlation with clinical symptoms and history are helpful in making the diagnosis.
CXRs are the most commonly performed diagnostic imaging study. A number of factors such as positioning of the patient and depth of inspiration can alter the appearance of the CXR, complicating interpretation further. In addition, clinicians are faced with reading high volumes of images every shift.
Pneumonia Detection
Now to detection Pneumonia we need to detect Inflammation of the lungs. In this project, you’re challenged to build an algorithm to detect a visual signal for pneumonia in medical images. Specifically, your algorithm needs to automatically locate lung opacities on chest radiographs. Business Domain Value Automating Pneumonia screening in chest radiographs, providing affected area details through bounding box. Assist physicians to make better clinical decisions or even replace human judgement in certain functional areas of healthcare (eg, radiology).
Guided by relevant clinical questions, powerful AI techniques can unlock clinically relevant information hidden in the massive amount of data, which in turn can assist clinical decision making.

### Project Description
In this capstone project, the goal is to build a pneumonia detection system, to locate the position of inflammation in an image.
Tissues with sparse material, such as lungs which are full of air, do not absorb the X-rays and appear black in the image. Dense tissues such as bones absorb X-rays and appear white in the image.
While we are theoretically detecting “lung opacities”, there are lung opacities that are not pneumonia related.
In the data, some of these are labeled “Not Normal No Lung Opacity”.
This extra third class indicates that while pneumonia was determined not to be present, there was nonetheless some type of abnormality on the image and oftentimes this finding may mimic the appearance of true pneumonia.
Dicom original images:- Medical images are stored in a special format called DICOM files (*.dcm). They contain a combination of header metadata as well as underlying raw image arrays for pixel data.

### Details about the data and dataset files are given in below link,
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

### 1: Pre-Processing, Data Visualisation, EDA and Model Building

● Exploring the given Data files, classes and images of different classes.

● Dealing with missing values

● Visualisation of different classes

● Analysis from the visualisation of different classes.

● Building a pneumonia detection model starting from basic CNN and then improving upon it.

● Train the model

● To deal with large training time, save the weights so that you can use them when training the model for the second time without starting from scratch. 

### 2: Test the Model, Fine-tuning and Repeat 

● Test the model and report as per evaluation metrics 

● Try different models 

● Set different hyper parameters, by trying different optimizers, loss functions, epochs, learning rate, batch size, checkpointing, early stopping etc..for these models to fine-tune them 

● Report evaluation metrics for these models along with your observation on how changing different hyper parameters leads to change in the final evaluation metric.

## Project Objectives
The objective of the project are,

● Learn to how to do build an Object Detection Model

● Use transfer learning to fine-tune a model. ● Learn to set the optimizers, loss functions, epochs, learning rate, batch size, checkpointing, early stopping etc.

● Read different research papers of given domain to obtain the knowledge of advanced models for the given problem.

## Reference
Acknowledgment for the datasets. https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview/acknowledgements



# Capstone Project: Pneumonia Detection using Chest X-Rays

## Table of Contents
- [Overview](#overview)
- [Problem Statement, Data, and Findings](#problem-statement-data-and-findings)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Dataset Information](#dataset-information)
  - [Data Merging](#data-merging)
  - [Class and Target Information](#class-and-target-information)
  - [Imbalance Class Problem](#imbalance-class-problem)
  - [EDA on DICOM Images Dataset](#eda-on-dicom-images-dataset)
  - [DICOM Images with Bounding Box Locating Lung Opacity](#dicom-images-with-bounding-box-locating-lung-opacity)
- [Data Pre-Processing](#data-pre-processing)
- [Model Building and Evaluation](#model-building-and-evaluation)
  - [Training and Testing Data Splits](#training-and-testing-data-splits)
  - [Convolutional Neural Network (Base Model)](#convolutional-neural-network-base-model)
  - [Transfer Learning Model](#transfer-learning-model)
- [Test Prediction and Final Submission Excel Sheet](#test-prediction-and-final-submission-excel-sheet)
  - [Predicting Result](#predicting-result)
  - [Submission](#submission)
- [Learning and Improvements](#learning-and-improvements)

## Overview
Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it. The infection causes inflammation in the air sacs in your lungs, which are called alveoli.
Pneumonia accounts for over 15% of all deaths of children under 5 years old internationally. In 2015, 920,000 children under the age of 5 died from the disease. 
While common, accurately diagnosing pneumonia is a tall order. It requires review of a chest radiograph (CXR) by highly trained specialists and confirmation through clinical history, vital signs and laboratory exams. 
Pneumonia usually manifests as an area or areas of increased opacity on CXR. However, the diagnosis of pneumonia on CXR is complicated because of a number of other conditions in the lungs such as fluid overload (pulmonary edema), bleeding, volume loss (atelectasis or collapse), lung cancer, or post-radiation or surgical changes. 
Outside of the lungs, fluid in the pleural space (pleural effusion) also appears as increased opacity on CXR. When available, comparison of CXRs of the patient taken at different time points and correlation with clinical symptoms and history are helpful in making the diagnosis.
CXRs are the most commonly performed diagnostic imaging study. A number of factors such as positioning of the patient and depth of inspiration can alter the appearance of the CXR, complicating interpretation further. 


Source: [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview) from Kaggle.

## Problem Statement, Data, and Findings
- The goal is to build a pneumonia detection system to locate the position of inflammation in an image. 

- While we are theoretically detecting “lung opacities”, there are lung opacities that are not pneumonia related. some of these are labelled “Not Normal No Lung Opacity”. 

- This extra third class indicates that while pneumonia was determined not to be present, there was nonetheless some type of abnormality on the image and oftentimes this finding may mimic the appearance of true pneumonia

### Data Source: 
RSNA Pneumonia Detection Challenge by Kaggle (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)

The data is organized in folders:

● The training data is provided as a set of patient Ids and bounding boxes. Bounding boxes are defined as follows: x-min, y-min, width, height

● There is also target column “ Target “ , indicating pneumonia or non-pneumonia.

● There may be multiple rows per patient Id

● All provided images are in DICOM format

  - stage_2_detailed_class_info.csv:  In class detailed info dataset are given the detailed information about the type of positive or negative class associated with a certain patient.
  - stage_2_train_labels.csv: In train labels dataset are given the patient ID and the window (x min, y min, width and height of the) containing evidence of pneumonia.
  - stage_2_train_images.zip: The directory containing 26000 training set raw image (DICOM) files
  - stage_2_test_images.zip: The directory containing 3000 testing set raw image (DICOM) files 

### DATASET INFORMATION:
  - Stage_2_detailed_class_info.csv dataset: Class info csv contains 30227 rows with 2 columns patient Id and class
 ![image](https://github.com/KrantiWalke/Computer-Vision-Projects/assets/72568005/414335db-f0b5-4caa-8ace-68d156254fc8)

  - Structure of stage_2_train_labels.csv dataset: train_labels.csv contains 30227 rows with columns
![image](https://github.com/KrantiWalke/Computer-Vision-Projects/assets/72568005/7069ee92-6e47-47ae-8c5a-4be16b78687d)

- DATA FIELD AND TYPES in the train_labels.csv: 
![image](https://github.com/KrantiWalke/Computer-Vision-Projects/assets/72568005/d5d3edba-a2a8-457e-8e9b-494fc17e0a18)


## Exploratory Data Analysis
### Dataset Information
The dataset comprises DICOM images and CSV files detailing patient IDs, class information, and bounding box coordinates for detected lung opacities.

### Data Merging
Data from multiple CSV files was combined into a single DataFrame to facilitate analysis.

### Class and Target Information
An analysis of class distribution showed an imbalance, necessitating techniques to address this issue in model training.

### Imbalance Class Problem
The class imbalance was evident in the dataset, with a higher number of non-pneumonia cases compared to pneumonia cases.

### EDA on DICOM Images Dataset
Exploratory analysis on DICOM images provided insights into the characteristics of pneumonia opacities and other lung conditions.

### DICOM Images with Bounding Box Locating Lung Opacity
Visualizations were created to demonstrate how bounding boxes can highlight areas of lung opacity in X-ray images.

## Data Pre-Processing
The preprocessing steps included resizing images for model input and addressing class imbalance by creating balanced datasets.

## Model Building and Evaluation
### Training and Testing Data Splits
The dataset was split into training and testing sets to evaluate model performance.

### Convolutional Neural Network (Base Model)
A CNN model was developed with layers designed to capture features relevant to pneumonia detection.

### Transfer Learning Model
A transfer learning approach was employed using MobileNet, combined with a UNET architecture for predicting bounding box coordinates.

## Test Prediction and Final Submission Excel Sheet
### Predicting Result
The final models were used to make predictions on the test set, identifying pneumonia presence and bounding box coordinates.

### Submission
Predictions were compiled into a submission format consistent with the competition requirements.

## Learning and Improvements
The project provided valuable experience in handling medical imaging data and building complex models for object detection. Future improvements could include utilizing more powerful computing resources to train deeper models for enhanced accuracy.

*Team Members: Dhruv Khanna, Harish, Saurav, Kranti, and Ragu.*

*Submission Date: 17/05/2020*

*Note: All code developed for this project is original and created by the team members.*

