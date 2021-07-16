# -*- coding: utf-8 -*-
"""
Created on Wed May 12 09:14:51 2021

@author: abc
"""

# PART : 1 Feature Extraction
#import library's
import numpy as np
import cv2
import pandas as pd

#read the image
img = cv2.imread('train_images\Sandstone_Versa0050.tif')
  
#Convert the image into gray level
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#reshape our image
img2 = img.reshape(-1) 

#Create dataframe
df = pd.DataFrame()

#Add original pixel values to the data frame as feature #1

df['Original Image'] = img2

#Add other features :
    
#First set - Gabor features
#Generate gabor features
num = 1 #To count numbers up in order to give gabor features a lable in the data frame
kernels = []
for theta in range(2):          #define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1,3) :    #sigma with 1 and 3
        for lamda in np.arange(0, np.pi / 4):  #Range of wavelengths
            for gamma in (0.05,0.5):   #Gamma values of 0.05 and 0.5
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.s
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
                #Now filter the image and add values to a new column
                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                print(gabor_label, ': theta=',theta, ': sigma=', sigma, ': lamda= ', lamda, ':gamma= ', gamma)
                num += 1 #Increment for gabor column Label


###############################################################################################################

#Second set - Canny edge
#Generate Canny edge

edges = cv2.Canny(img, 100, 200)
edges1 = edges.reshape(-1)
df['Canny Edge'] = edges1

#print(df.head())
######################################################################################

#Third set - Roberts, sobel, scharr, prewitt, gaussian, median filter
from skimage.filters import roberts, sobel, scharr, prewitt

#Roberts filter
edge_roberts = roberts(img)
edge_roberts1 = edge_roberts.reshape(-1)
df['Roberts'] = edge_roberts1

#Sobel filter
edge_sobel =sobel(img)
edge_sobel1 = edge_sobel.reshape(-1)
df['Sobel'] = edge_sobel1

#scharr filter
edge_scharr = scharr(img)
edge_scharr1 = edge_scharr.reshape(-1)
df['Scharr'] = edge_scharr1

#Prewitt filter
edge_prewitt = prewitt(img)
edge_prewitt1 = edge_prewitt.reshape(-1)
df['Prewitt'] = edge_prewitt1

#GAUSSIAN with sigma=3
from scipy import ndimage as nd
gaussian_img = nd.gaussian_filter(img, sigma=3)
gaussian_img1 = gaussian_img.reshape(-1)
df['Gaussian s3'] = gaussian_img1

#Gaussian with sigma = 7
gaussian_img2 = nd.gaussian_filter(img, sigma=7)
gaussian_img3 = gaussian_img2.reshape(-1)
df['Gaussian s7'] = gaussian_img3

#MEDIAN with size = 3
median_img = nd.median_filter(img, size = 3)
median_img1 = median_img.reshape(-1)
df['Median s3'] = median_img1

#VARIANCE with size = 3
#variance_img = nd.generic_filter(img, np.var, size = 3)
#variance_img1 = variance_img.reshape(-1)
#df['Variance s3'] = variance_img1   

#print(df.head())

#Add one column that corresponds to image

labeled_img = cv2.imread('train_masks\Sandstone_Versa0050.tif')

#Convert labeled image into gray image
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
#Reshape labeled image
labeled_img1 = labeled_img.reshape(-1)
#create one dataframe for labeled image
df['Labels'] = labeled_img1

#print(df.head())

############################################################################


#PART : 2 Train a model 

# STEP : 1 define dependent and independent variables

#Dependent variable
Y = df['Labels'].values

#Independent variable
X = df.drop(labels = ['Labels'], axis=1) #keep everything expect for label


# STEP : 2 split our above data into training and testing 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

#test_size = 0.4 means in above hall data we use 40% for testing purpose
#random_state = 20 means it give 20 random values like that they didn't change while execution


# STEP : 3 Import ML algorithm and train a model
#USING RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

#create an instance for the model
model = RandomForestClassifier(n_estimators = 100, random_state = 42)

#fit our data
model.fit(X_train, Y_train)

#Predict our traning data
prediction_test = model.predict(X_test)

#Find Accuracy of iour predicted data

from sklearn import metrics
print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))











 


























