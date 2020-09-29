from skimage import measure
import pandas as pd
from skimage.metrics import structural_similarity
from cv2 import cv2
from sklearn.metrics.cluster import entropy
import numpy as np
import os
file={}
rules=['mean','max','min']
directory=('C:/Users/dreddyyerram/Desktop/imagefusion/Set1_10')
for i in os.listdir(directory):
    
    temp=[]
    for j in os.listdir(directory+"/"+i):
        
        temp.append(j)
    file[i]=temp

def cross_entropy(x, y):


    if np.any(x < 0) or np.any(y < 0):
        raise ValueError('Negative values exist.')


    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)
    x /= np.sum(x)
    y /= np.sum(y)

    # Ignore zero 'y' elements.
    mask = y > 0
    x = x[mask]
    y = y[mask]
    ce = -np.sum(x * np.log(y))
    return ce



for ite in file.keys():

    fusionmethod = 'max'
    input1 = cv2.imread('Set1_10/'+ite+'/'+file[ite][0],0)
    input2 = cv2.imread('Set1_10/'+ite+'/'+file[ite][1],0)
    input1 = cv2.resize(input1,(512,512))
    input2 = cv2.resize(input2,(512,512)) 




    type=[]
    Rule_for_ll=[]
    Rule_for_rest=[]
    CrossEntropy=[]
    Entropy=[]
    SSIM_with_CT=[]
    SSIM_with_MR=[]
    #metrics for LE
    stdresults={}
    std=[]
    entropyresults={}
    crossentropyresults={}
    SSIM={}
    """for j in rules:
        for k in rules:
            if j!=k:
                
                name = "fusedimages/"+ite+"/LE_" + j + "(LL)_" + k + "(rest).jpg"
                output = cv2.imread(name,0)
                output=cv2.resize(output,(512,512))
                s1 = structural_similarity(input1, output,multichannel=True)
                s2= structural_similarity(input2, output,multichannel=True)
                type.append("Local Energy")
                Rule_for_ll.append(j)
                Rule_for_rest.append(k)
                SSIM_with_CT.append(s1)
                SSIM_with_MR.append(s2)
                SSIM[name]=(s1,s2)
                entropyresults[name] = entropy(output)
                Entropy.append(entropyresults[name])
                crossentropyresults[name] = (cross_entropy(input1, output)+cross_entropy(input2, output))/2
                CrossEntropy.append(crossentropyresults[name])
                stdresults[name]=output.std()
                std.append(stdresults[name])"""
    for j in rules:
        for k in rules:
            if j!=k:
               
                name = "fusedimages/"+ite+"/NOR_" + j + "(LL)_" + k + "(rest).jpg"
                output = cv2.imread(name,0)
                
                output=cv2.resize(output,(512,512))
                s1 = structural_similarity(input1, output,multichannel=True)
                s2= structural_similarity(input2, output,multichannel=True)
                type.append("Normal")
                Rule_for_ll.append(j)
                Rule_for_rest.append(k)
                SSIM_with_CT.append(s1)
                SSIM_with_MR.append(s2)
                SSIM[name]=(s1,s2)
                entropyresults[name] = entropy(output)
                Entropy.append(entropyresults[name])
                crossentropyresults[name] = (cross_entropy(input1, output)+cross_entropy(input2, output))/2
                CrossEntropy.append(crossentropyresults[name])
                stdresults[name]=output.std()
                std.append(stdresults[name])

    
    name = "fusedimages/"+ite+"/LE_"+ite.replace(' ', '_')+".jpg"
    
    print(name)
    
    output = cv2.imread(name,0)
    print(output)
    output=cv2.resize(output,(512,512))
    s1 = structural_similarity(input1, output,multichannel=True)
    s2= structural_similarity(input2, output,multichannel=True)
    type.append("LE")
    Rule_for_ll.append("max")
    Rule_for_rest.append("min")
    SSIM_with_CT.append(s1)
    SSIM_with_MR.append(s2)
    SSIM[name]=(s1,s2)
    entropyresults[name] = entropy(output)
    Entropy.append(entropyresults[name])
    crossentropyresults[name] = (cross_entropy(input1, output)+cross_entropy(input2, output))/2
    CrossEntropy.append(crossentropyresults[name])
    stdresults[name]=output.std()
    std.append(stdresults[name])
    dict = {'type': type, 'Rule_for_ll': Rule_for_ll, 'Rule_for_rest': Rule_for_rest,'Entropy':Entropy,'SSIM_with_CT' : SSIM_with_CT,'SSIM_with_MR': SSIM_with_MR, 'Cross Entropy':CrossEntropy, "standard deviation" : std}
    df = pd.DataFrame(dict)
    df.to_csv("fusedimages/"+ite+"/metric"+ite+".csv")
    print(df)
    print(entropyresults)
    print(SSIM)
    print(std)







