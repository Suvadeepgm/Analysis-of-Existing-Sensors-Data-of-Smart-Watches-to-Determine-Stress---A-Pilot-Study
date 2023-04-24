# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:31:13 2023

@author: suvadeep
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split 
import seaborn as sns
import glob
import os
import warnings
warnings.filterwarnings('ignore')
import scipy.signal as signal
from scipy.signal import find_peaks
import biosppy.signals.bvp as bvp
import entropy_estimators as ee
import biosppy.signals.resp as resp
import biosppy.signals.ecg as ecg
import biosppy.signals.tools as st
import peakutils
import plotly.graph_objs as go
import nolds
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


#Function to add time stamp to all the features reading based on starting time and frequency
def addTimeStamp(df,start,freq):  
    Timestamp=[]
    i=0
    while (i<len(df)):
        for j in range(freq): 
            if i>=len(df):
                break;
            Timestamp.append(start)
            i+=1
        start+=1
    return Timestamp


#Function to estimate Respiratory Rate based on peaks of BVP Signal
def BVP_to_RR(hyperdata):
                fs=1.1
                # Remove baseline wander using a high-pass filter
                fc_hp = 0.5  # High-pass filter cutoff frequency in Hz
                b, a = signal.butter(2, fc_hp / (fs / 2), 'highpass')
                bvp_hp = signal.filtfilt(b, a, hyperdata)


                # Remove noise and extract frequency range using a bandpass filter
                fc_bp = [0.1, 0.5]  # Bandpass filter cutoff frequency in Hz
                b, a = signal.butter(2, fc_bp, 'bandpass')
                bvp_bp = signal.filtfilt(b, a, bvp_hp)
                #print(bvp_bp)

                dbvp_signal = np.diff(bvp_bp)

                # Find the peaks in the derivative signal
                peaks, _ = find_peaks(dbvp_signal, height=0)


                # Compute the inter-peak intervals (in seconds)
                ipi = np.diff(peaks) / fs
                #print(np.mean(ipi))
                # Compute the respiratory rate (in breaths per minute)
                rr = 60 / np.mean(ipi)
                return rr
            
            

###############################################################################
#####################  Random Forest Model ####################################
#Random Forest Classifier
def randomForest(df,shuffle):
    df= "D:\\EssexFiles\\DSDM\\Stress-Predict-Dataset-main\\Stress-Predict-Dataset-main\\Data_Processed\\" +df+ ".csv"
    df= pd.read_csv(df)
    X = df.drop(columns=['Stress','Timestamp','Accelerometer_X','Accelerometer_Y','Accelerometer_Z'], axis = 1)
    y = df['Stress'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=shuffle) 

    rfc = RandomForestClassifier(n_estimators=100)

    # Fit the model on the training data
    rfc.fit(X_train, y_train)

    # Predict on the testing data
    y_pred = rfc.predict(X_test)

    # Generate a classification report
    report = classification_report(y_test, y_pred)
    #print(report)

    cm = confusion_matrix(y_test, y_pred)
   #print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    #print('Accuracy:', accuracy)

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, cmap="Blues")
    #plt.show()
    return accuracy 

    
################################################################################
######################## LSTM Model  ###########################################

#LSTM Model
def lstm(df,shuffle):
    df= "D:\\EssexFiles\\DSDM\\Stress-Predict-Dataset-main\\Stress-Predict-Dataset-main\\Data_Processed\\" +df+ ".csv"
    df= pd.read_csv(df)
    X = df.drop(columns=['Stress','Timestamp','Accelerometer_X','Accelerometer_Y','Accelerometer_Z'], axis = 1)
    y = df['Stress'] 

    model = Sequential()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
    timesteps = 30 # assuming a sequence length of 30

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(64,return_sequences=True,  dropout=0.2, recurrent_dropout=0.2, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(32,return_sequences=True, ))
    model.add(LSTM(8, return_sequences=True))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    # evaluate the model on the test data
    score = model.evaluate(X_test, y_test, batch_size=32)
    #print(score)

    y_pred = model.predict(X_test)
    # evaluate the model performance
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = np.round(y_pred)
    return accuracy
                
################################################################################            
########################################## Loading the Data ####################


#path of raw data
input_path = 'D:\EssexFiles\DSDM\Stress-Predict-Dataset-main\Stress-Predict-Dataset-main\Raw_data' 
#path of feature specific files for individual paticipant combined with timestamp for each data 
temp_path= 'D:\EssexFiles\DSDM\Stress-Predict-Dataset-main\Stress-Predict-Dataset-main\Temp\\'  
#path of data of individual paticipant with features joined on Timestamp
output_path = 'D:\EssexFiles\DSDM\Stress-Predict-Dataset-main\Stress-Predict-Dataset-main\Data_Processed\\'
tempRR='D:\EssexFiles\DSDM\Stress-Predict-Dataset-main\Stress-Predict-Dataset-main\TempRR\\' 

#creating a list of file paths present inside the input path
data_process = glob.glob(os.path.join(input_path, '**/*.csv'))  



for i in data_process:
    x = i.split('\\')
    
    #getting the participant number/sample number
    sampleName=x[len(x)-2]  
    x=x[len(x)-1].split(".")
    
    #getting the feature name
    fileName=x[0] 
    
    ## IBI data is disregarded as it is derived from the BVP data. Time tags are also not considered for timestamp entry
    if fileName=='IBI' or fileName=="tags_"+sampleName: 
        continue  
   
    colName=fileName+"_"+sampleName
    file_all=pd.read_csv(i,header=None, delimiter=",")
    colcount=len(file_all.index)
    #getting the initial start time of the session
    file_start= int(file_all.loc[0,0]) 
    #getting the frequency of the observations
    file_freq = int(file_all.loc[1,0])
    #getting the entire data frame without the first two rows  
    file_df=file_all.iloc[2:] 
    
    #Invoking the addTimeStamp function to get the timestamp value for each data  
    file_df["Timestamp"]=addTimeStamp(file_df,file_start,file_freq)  
    
    #ACC data column are named properly based on Axis    
    if fileName=='ACC':
        file_df.rename(columns={ file_df.columns[0]: "Accelerometer_X"}, inplace = True) 
        file_df.rename(columns={ file_df.columns[1]: "Accelerometer_Y"}, inplace = True) 
        file_df.rename(columns={ file_df.columns[2]: "Accelerometer_Z"}, inplace = True) 
    else:
    #Rest of the data columns are named properly based on Filename    
        file_df.rename(columns={ file_df.columns[0]: fileName }, inplace = True) 
    
    #Taking the mean of data which have same timestamp  
    file_df_mean=file_df.groupby(['Timestamp']).mean() 
    
    #####################################################
    for i in range(0, colcount%file_freq):
        file_df_mean.drop(index=file_df_mean.index[-1], inplace=True)
 
    #####################################################
    outputPath= temp_path+ sampleName +"\\" 
    if os.path.exists(outputPath) == False: 
        os.mkdir(outputPath) 
    outputPath= tempRR+ sampleName +"\\" 
    if os.path.exists(outputPath) == False: 
        os.mkdir(outputPath)
    colNameTemp="BVP_"+sampleName
    if colName==colNameTemp:
        file_df.to_csv(tempRR+ sampleName +"\\"+ colName +"_Time.csv" , encoding='utf-8', index=True)
    file_df_mean.to_csv(temp_path+ sampleName +"\\"+ colName +".csv" , encoding='utf-8', index=True)
    
    
    
    
    
###########################################################################################

temp_process = glob.glob(os.path.join(tempRR, '**/*.csv')) 
hyperdata=[]
rrstore=[]
respRate=0
fs=1.1
p=1
for i in temp_process: #Iterating participants
    x = i.split('\\')
     
    #getting the participant number/sample number
    sampleName=x[len(x)-2]  
    x=x[len(x)-1].split(".")
    s=1
    #getting the feature name
    fileName=x[0] 
    filenameTemp="BVP_"+sampleName+"_Time"
    ## IBI data is disregarded as it is derived from the BVP data. Time tags are also not considered for timestamp entry
    if fileName==filenameTemp: 
        file_all=pd.read_csv(i, delimiter=",")
        start= int(file_all.iloc[1,2])
        #print(start)
        for z in range(1,len(file_all)):
            #print(file_all.iloc[z-1,2])
            if file_all.iloc[z-1,2]==start:
                hyperdata.append(file_all.iloc[z-1,1])
                #print(hyperdata)
            else:
                respRate = BVP_to_RR(hyperdata)
                rrstore.append(respRate)
                start+=1
                hyperdata=[]
                #print("Done for S",s)
                s+=1
                
       ##########output path for Respiratory Rate#######
        outputPath= tempRR+ sampleName +"_RR\\" #path for rr storage
        if os.path.exists(outputPath) == False: 
            os.mkdir(outputPath)
        colNameTemp="BVP_"+sampleName
        
        sample= pd.read_csv(temp_path+ sampleName +"\\"+ colNameTemp +".csv", delimiter=",") 
        #########################
        rrdf=pd.DataFrame(rrstore)
        sample=pd.concat([sample, rrdf], axis=1) 
        sample.to_csv(temp_path+ sampleName +"\\"+ colNameTemp +".csv" , encoding='utf-8', index=False)
        rrstore=[] 
        print("Respiratory Rate estimated for sample ",p)
        p+=1
        
############################################################################################
#######################      Joining and Labelling the Data for each participant
############################################################################################

for i in range (1,36):
    if i<10:
        sampleNumber="S0"+str(i)
    else:
        sampleNumber="S"+str(i)
    data_join = glob.glob(os.path.join(temp_path, sampleNumber+'/*.csv')) 
    #print(data_join)
    #Joining the features of each invidual - Outer join on Timestamp   
    df1=pd.read_csv(data_join[0], delimiter=",")
    df2=pd.read_csv(data_join[1], delimiter=",") 
    df2 = df2.rename(columns={df2.columns[2]: 'RR'})
    df3=pd.read_csv(data_join[2], delimiter=",")
    df4=pd.read_csv(data_join[3], delimiter=",")
    df5=pd.read_csv(data_join[4], delimiter=",")
    
    
        
    S01a = pd.merge(df1,df2, on=["Timestamp"], how="outer")
    S01b= pd.merge(S01a,df3, on=["Timestamp"], how="outer")
    S01c= pd.merge(S01b,df4, on=["Timestamp"], how="outer")
    S01= pd.merge(S01c,df5, on=["Timestamp"], how="outer")
    print("##.....Null Value Count for "+sampleNumber+"....##")
    print(S01.isnull().sum(axis = 0))
    
    
    #Dropping the rows with null values  
    S01.dropna(inplace=True)
    
    time=pd.read_csv(input_path+"\\"+sampleNumber+"\\tags_"+sampleNumber+".csv",header=None)
    #timeList=time.values.tolist()
    
    outputPath= output_path+ sampleName +"\\"
    
    #Taking the Timestamp for each individual and calculating the stress level based on time tags 
    #Need to do more research on decising factors for stress level
    #Stress is labelled as 1 during Stroop Test, during Interview and during Hyperventillation test 
    
    df=S01['Timestamp']
    temp=df.values.tolist()
    stress=[]
    
    #Labelling the stress column based on the timestamps of the events
    for z in range(0,len(temp)):
        if temp[z]>time.loc[0,0] and temp[z]<time.loc[1,0]:
            stress.append(1)
        elif temp[z]>time.loc[2,0] and temp[z]<time.loc[3,0]:
            stress.append(1)
        elif temp[z]>time.loc[4,0] and temp[z]<time.loc[5,0]:
            stress.append(1)
        elif temp[z]>time.loc[6,0] and temp[z]<time.loc[6,0]+300:
            stress.append(1)
        else:
            stress.append(0) 
       
    S01["Stress"]=stress
    S01.to_csv(output_path+ sampleNumber +".csv" , encoding='utf-8', index=False)
    
    
    
###############################################################################
############################  Model Fit #######################################
rfc=[]
lst=[]
for i in range (1,36):
    if i<10:
        sampleNumber="S0"+str(i)
    else:
        sampleNumber="S"+str(i)
    r=randomForest(sampleNumber,False)
    print("Random Forest Accuracy for "+sampleNumber, r)
    rfc.append(r)
    l=lstm(sampleNumber,False)
    print("LSTM Accuracy for "+sampleNumber,l)
    lst.append(l) 
    
    
maxRFC=max(rfc)
maxLSTM=max(lst)
print("Maximum accuracy of Random Forest Classifier: ",maxRFC)
print("Maximum accuracy of Long Short Term Memory: ",maxLSTM)




