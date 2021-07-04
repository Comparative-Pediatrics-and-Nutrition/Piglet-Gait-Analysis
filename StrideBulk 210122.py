# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:12:52 2020

@author: jdr248
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import diff
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.ndimage import median_filter


def distance_numpy(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)

def StepItUp3(foot,df,prms,diists, wees, A,B,C,D,W):
    if foot == 'Foot_Right_Front_x':
        yFoot = 'Foot_Right_Front_y'
        fFoot = 'dRF'
    elif foot == 'Foot_Left_Front_x':
        yFoot = 'Foot_Left_Front_y'
        fFoot = 'dLF'
    elif foot == 'Foot_Right_Back_x':
        yFoot = 'Foot_Right_Back_y'
        fFoot = 'dRB'
    elif foot == 'Foot_Left_Back_x':
        yFoot = 'Foot_Left_Back_y'
        fFoot = 'dLB'
    smooth = median_filter(df[fFoot],5)
    peaks, properties = find_peaks(smooth, prominence= prms,distance = diists,width=wees)
    peaks = df.index[0] + peaks
    peaks = np.append(peaks,df.index[-1])
    peaks = np.insert(peaks,0,df.index[0])

    StepLengths = np.empty(len(peaks)-2)
    for steps in range(len(peaks)-2):
        Av1 = df.loc[df.index>peaks[steps]]
        Av1 = Av1.loc[Av1.index<peaks[steps+1]]
        Av1 = Av1.loc[Av1[fFoot]<3]
        Av1x = Av1[foot].mean()
        Av1y = Av1[yFoot].mean()
        P1 = np.array([Av1x,Av1y])  
        
        Av2 = df.loc[df.index>peaks[steps+1]]
        Av2 = Av2.loc[Av2.index<peaks[steps+2]]
        Av2 = Av2.loc[Av2[fFoot]<3]
        Av2x = Av2[foot].mean()
        Av2y = Av2[yFoot].mean()
        P2 = np.array([Av2x,Av2y])
        
        
        P1dN = distance_numpy(A,D,P1)
        P1dS = distance_numpy(B,C,P1)
        
        P1dE = distance_numpy(A,B,P1)
        P1dW = distance_numpy(D,C,P1)
        
        P1newY = P1dN/(P1dN+P1dS)*W
        P1newX = P1dE/(P1dE+P1dW)*180
        
        P1new = np.array([P1newX,P1newY])
        
        P2dN = distance_numpy(A,D,P2)
        P2dS = distance_numpy(B,C,P2)
        
        P2dE = distance_numpy(A,B,P2)
        P2dW = distance_numpy(D,C,P2)
        
        P2newY = P2dN/(P2dN+P2dS)*W
        P2newX = P2dE/(P2dE+P2dW)*180
        
        P2new = np.array([P2newX,P2newY])
                   
        if np.isnan(np.sum(P2new)):
            continue
        if np.isnan(np.sum(P1new)):
            continue
        StepSize = distance.euclidean(P1new,P2new)
        StepLengths[steps] = StepSize
    StepLengths = StepLengths[StepLengths>6]   
    
    swing = properties['widths']

    stance = np.empty(len(peaks)-3)
    
    for steps in range(1,len(peaks)-2):
        second = peaks[steps+1]-swing[steps]/2
        first = peaks[steps]-swing[steps-1]/2
        stance[steps-1]= second-first
    return StepLengths,swing,stance




# path = "/home/lats/Downloads/Pigs/Examples/*.csv"
path = "Q:\Common\Lab member folder\Matt\AI\Pigs\Hitters\*.csv"
#path = r"Q:\Common\Lab member folder\Matt\AI\Pigs\New\*.csv"
# path = "Q:\Common\Lab member folder\Matt\AI\Pigs\Hitters\*.csv"
Total_Data = pd.DataFrame()

for file in glob.glob(path):
    print (file)
    Data = pd.read_csv(file,delimiter = ',')
    Data.columns = Data.iloc[0]+'_'+Data.iloc[1]
    Data = Data.drop([Data.index[0] , Data.index[1]])
    Data=Data.apply(pd.to_numeric, errors='coerce')
    Data.reset_index(drop=True, inplace=True)
    
    C_Data = Data.copy()
    C_Col = C_Data.columns
    Like_Col = [i for i in C_Col if "likelihood" in i] 
    x_Col = [i for i in C_Col if "_x" in i] 
    y_Col = [i for i in C_Col if "_y" in i]
    
    Threshold = 0.8
    for count in range(C_Data.index.size): #for each row
        for col in Like_Col:                # for each liklehood
            colID = C_Data.columns.get_loc(col) #get the column number for indexing
            if C_Data.iloc[count,colID] < Threshold:  #if liklehood is to low
                C_Data.iloc[count,colID-2] = np.nan
                C_Data.iloc[count,colID-1] = np.nan

    C_Data = C_Data.interpolate(method='linear',axis = 0,limit=20)
    C_Data = C_Data.fillna(method='bfill',axis = 0,limit=20)
    
    dy = diff(ndimage.median_filter(C_Data.loc[ : , 'Nose_x' ], size=15))
    dy2 = diff(ndimage.median_filter(C_Data.loc[ : , 'Shoulder_x' ], size=15))
    move = (dy+dy2)/2
    move2=np.append(move,0)
    C_Data.insert(1,'Move',move2)
    C_Data.loc[ : ,'Move' ] = ndimage.median_filter(C_Data.loc[ : , 'Move' ], size=15)
    C_Data['Move'] = C_Data['Move'].rolling(15,min_periods=1).mean()
    
    peaks, properties = find_peaks(C_Data.Move, width=20,prominence= 2,  height=(1, 50))

    s0 = [0,0]
    for Ps in range(len(peaks)):
        size = properties.get('peak_heights')[Ps]*properties.get('widths')[Ps]
        if size > s0[0]:
            s0[0] = size
            s0[1] = Ps
    Threshold = 0.5        
    a1 = C_Data.loc[peaks[s0[1]]:,'Move']
    events1 = np.split(a1, np.where(np.isnan(a1))[0])[0]
    forwardL = next((x1 for x1 in events1 if x1 < Threshold),-1)
    a2 = C_Data.loc[:peaks[s0[1]],'Move'][::-1]
    events2 = np.split(a2, np.where(np.isnan(a2))[0])[0]
    backwardL = next((x2 for x2 in events2 if x2 < Threshold),-1)
    if forwardL == -1 :
        fLI = properties.get('right_bases')[s0[1]]
    else:
        fLI = C_Data.loc[C_Data['Move'] == forwardL].index[0]
    if backwardL == -1 :
        bLI = properties.get('left_bases')[s0[1]]
    else:
        bLI = C_Data.loc[C_Data['Move'] == backwardL].index[0]


    D_Data =  C_Data.iloc[bLI:fLI,:]
    
    dRB = diff(D_Data.Foot_Right_Back_x)
    dRB=np.append(dRB,np.NaN)
    
    dLB = diff(D_Data.Foot_Left_Back_x)
    dLB=np.append(dLB,np.NaN)
    
    dRF = diff(D_Data.Foot_Right_Front_x)
    dRF=np.append(dRF,np.NaN)
    
    dLF = diff(D_Data.Foot_Left_Front_x)
    dLF=np.append(dLF,np.NaN)
    
    D_Data.insert(1,'dRB',dRB)
    D_Data.insert(1,'dLB',dLB)
    D_Data.insert(1,'dRF',dRF)
    D_Data.insert(1,'dLF',dLF)

    F_coutn = int(file.split("PIC_0",1)[1][0:3])
    if F_coutn <=362:
        A = np.array([173,514])
        B = np.array([135,564])
        C = np.array([1678,531])
        D = np.array([1637,488])
        W = 11.5
    elif F_coutn <=386:
        A = np.array([168,524])
        B = np.array([112,575])
        C = np.array([1690,555])
        D = np.array([1649,505])
        W = 13.5
    elif F_coutn <=424:
        A = np.array([203,507])
        B = np.array([160,558])
        C = np.array([1713,548])
        D = np.array([1680,503])
        W = 11.5
    elif F_coutn <=467:
        A = np.array([199,503])
        B = np.array([153,560])
        C = np.array([1709,548])
        D = np.array([1666,494])
        W = 13.5
    elif F_coutn <=482:
        A = np.array([207,487])
        B = np.array([150,558])
        C = np.array([1715,552])
        D = np.array([1661,489])
        W = 16
        
        
    stepFL,swingFL,stanceFL = StepItUp3('Foot_Left_Front_x',D_Data,8,10,1,A,B,C,D,W)
    stepFR,swingFR,stanceFR = StepItUp3('Foot_Right_Front_x',D_Data,8,10,1,A,B,C,D,W)
    stepBL,swingBL,stanceBL = StepItUp3('Foot_Left_Back_x',D_Data,8,10,1,A,B,C,D,W)
    stepBR,swingBR,stanceBR = StepItUp3('Foot_Right_Back_x',D_Data,8,10,1,A,B,C,D,W)
    
    H_all = np.sqrt(np.square(D_Data.Foot_Right_Front_x-D_Data.Shoulder_x)+np.square(D_Data.Foot_Right_Front_y-D_Data.Shoulder_y))
    ONEeighty = np.sqrt(np.square(C[0]-B[0])+np.square(C[1]-B[1]))
    
    H = 180*H_all.mean()/ONEeighty
    
    Distance = np.nanmean([D_Data['Nose_x'].iat[-1],D_Data['Shoulder_x'].iat[-1]])-np.nanmean([D_Data['Nose_x'].iat[0],D_Data['Shoulder_x'].iat[0]])
    Distance = 180*Distance/ONEeighty
    Time = len(D_Data)/30
    Speed = Distance / Time
    
    Total_Data = Total_Data.append({'Video': F_coutn,'Distance':Distance,'Time':Time,'Speed':Speed,'Step_Back_Left': stepBL,'Step_Back_Right': stepBR,'Step_Front_Left': stepFL,'Step_Front_Right': stepFR,'Swing_Back_Left': swingBL,'Swing_Back_Right': swingBR,'Swing_Front_Left': swingFL,'Swing_Front_Right': swingFR,'Stance_Back_Left': stanceBL,'Stance_Back_Right': stanceBR,'Stance_Front_Left': stanceFL,'Stance_Front_Right': stanceFR,'Height': H }, ignore_index=True)
    





#%%
average = lambda x: x.mean()

Total_Data['Stride_B_L_avg'] = Total_Data['Step_Back_Left'].apply(average)
Total_Data['Stride_B_R_avg'] = Total_Data['Step_Back_Right'].apply(average)
Total_Data['Stride_F_L_avg'] = Total_Data['Step_Front_Left'].apply(average)
Total_Data['Stride_F_R_avg'] = Total_Data['Step_Front_Right'].apply(average)

Total_Data['Stance_B_L_avg'] = Total_Data['Stance_Back_Left'].apply(average)/30
Total_Data['Stance_B_R_avg'] = Total_Data['Stance_Back_Right'].apply(average)/30
Total_Data['Stance_F_L_avg'] = Total_Data['Stance_Front_Left'].apply(average)/30
Total_Data['Stance_F_R_avg'] = Total_Data['Stance_Front_Right'].apply(average)/30

Total_Data['Swing_B_L_avg'] = Total_Data['Swing_Back_Left'].apply(average)/30
Total_Data['Swing_B_R_avg'] = Total_Data['Swing_Back_Right'].apply(average)/30
Total_Data['Swing_F_L_avg'] = Total_Data['Swing_Front_Left'].apply(average)/30
Total_Data['Swing_F_R_avg'] = Total_Data['Swing_Front_Right'].apply(average)/30



Total_Data['Stride_avg']= Total_Data[['Stride_B_L_avg','Stride_B_R_avg','Stride_F_L_avg','Stride_F_R_avg']].mean(axis=1)
Total_Data['Stance_avg']= Total_Data[['Stance_B_L_avg','Stance_B_R_avg','Stance_F_L_avg','Stance_F_R_avg']].mean(axis=1)
Total_Data['Swing_avg']= Total_Data[['Swing_B_L_avg','Swing_B_R_avg','Swing_F_L_avg','Swing_F_R_avg']].mean(axis=1)


Total_Data['Stride_avg_norm']=Total_Data['Stride_avg']/Total_Data['Height']
Total_Data['Stance_avg_norm']=Total_Data['Stance_avg']/ (980.665*np.sqrt(Total_Data['Height']))
Total_Data['Swing_avg_norm']=Total_Data['Swing_avg']/ (980.665*np.sqrt(Total_Data['Height']))

Total_Data['Stride_avg_norm_2']=Total_Data['Stride_avg']/Total_Data['Height']
Total_Data['Stance_avg_norm_2']=Total_Data['Stance_avg']/Total_Data['Height']
Total_Data['Swing_avg_norm_2']=Total_Data['Swing_avg']/Total_Data['Height']

Total_Data['Stride_avg_norm_3']=Total_Data['Stride_avg']/Total_Data['Height']
Total_Data['Stance_avg_norm_3']=Total_Data['Stance_avg']/(np.sqrt(Total_Data['Height'])/980.665)
Total_Data['Swing_avg_norm_3']=Total_Data['Swing_avg']/(np.sqrt(Total_Data['Height'])/980.665)

Total_Data['Stance_avg_norm_speed']=Total_Data['Stance_avg']/ (Total_Data['Speed'])
Total_Data['Swing_avg_norm_speed']=Total_Data['Swing_avg']/ (Total_Data['Speed'])


# Total_Data['AI_Stride']= 100*(Total_Data[['Stride_B_R_avg','Stride_F_R_avg']].mean(axis=1)-Total_Data[['Stride_B_L_avg','Stride_F_L_avg']].mean(axis=1))/(0.5*(Total_Data[['Stride_B_R_avg','Stride_F_R_avg']].mean(axis=1)+Total_Data[['Stride_B_L_avg','Stride_F_L_avg']].mean(axis=1)))
# Total_Data['AI_Stance']= 100*(Total_Data[['Stance_B_R_avg','Stance_F_R_avg']].mean(axis=1)-Total_Data[['Stance_B_L_avg','Stance_F_L_avg']].mean(axis=1))/(0.5*(Total_Data[['Stance_B_R_avg','Stance_F_R_avg']].mean(axis=1)+Total_Data[['Stance_B_L_avg','Stance_F_L_avg']].mean(axis=1)))
# Total_Data['AI_Swing']= 100*(Total_Data[['Swing_B_R_avg','Swing_F_R_avg']].mean(axis=1)-Total_Data[['Swing_B_L_avg','Swing_F_L_avg']].mean(axis=1))/(0.5*(Total_Data[['Swing_B_R_avg','Swing_F_R_avg']].mean(axis=1)+Total_Data[['Swing_B_L_avg','Swing_F_L_avg']].mean(axis=1)))






#%%
Total_Data.to_csv('Total_Data_210212.csv')









#%%
#Slice = Total_Data[count:count+3]
# Slice = Total_Data[121:123]
# print(Slice['Video'])

# BL = Slice['Back_Left'].apply(pd.Series).stack()
# BL = BL.reset_index()
# BL_Avg = BL.iloc[:,2].mean()

# BR = Slice['Back_Right'].apply(pd.Series).stack()
# BR = BR.reset_index()
# BR_Avg = BR.iloc[:,2].mean()

# FL = Slice['Front_Left'].apply(pd.Series).stack()
# FL = FL.reset_index()
# FL_Avg = FL.iloc[:,2].mean()

# FR = Slice['Front_Right'].apply(pd.Series).stack()
# FR = FR.reset_index()
# FR_Avg = FR.iloc[:,2].mean()

# frames = [BL.iloc[:,2],BR.iloc[:,2],FL.iloc[:,2],FR.iloc[:,2]]
# Steps_tot = pd.concat(frames)
# Steps_Avg = Steps_tot.mean()

# print(BL_Avg,BR_Avg,FL_Avg,FR_Avg,Steps_Avg)
# count = count+3