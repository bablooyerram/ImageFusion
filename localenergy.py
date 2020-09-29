import pywt
from cv2 import cv2
import numpy as np
import os
file={}
directory=('C:/Users/dreddyyerram/Desktop/imagefusion/Set1_10')
for i in os.listdir(directory):
    
    temp=[]
    for j in os.listdir(directory+"/"+i):
        
        temp.append(j)
    file[i]=temp

rules=['mean','max','min']

def fuseCoeff(cooef1, cooef2, method):
    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    else:
        cooef = []
    return cooef


def localenergy(coeff):
    a = np.asarray(coeff[0])
    m, n = a.shape
    I1LE = np.copy(a)
    neighb = [-1, 0, 1]
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            sum = 0
            for k in neighb:
                for l in neighb:
                    if k != 0 or l != 0:
                        sum = sum + (a[i + k][j + l] * a[i + k][j + l])
            I1LE[i][j] = sum

    edited_coeff = (I1LE, coeff[1])
    return edited_coeff



for ite in file.keys():

    fusionmethod = 'max'
    I1 = cv2.imread('Set1_10/'+ite+'/'+file[ite][0],0)
    I2 = cv2.imread('Set1_10/'+ite+'/'+file[ite][1],0)
   
    I1 = cv2.resize(I1,(512,512))
    I2 = cv2.resize(I2,(512,512))   
    wavelet = 'db1'
    cooef1 = pywt.dwt2(I1[:,:], wavelet)
    cooef2 = pywt.dwt2(I2[:,:], wavelet)
    print(ite)
    LE_cooef1=localenergy(cooef1)
    LE_cooef2=localenergy(cooef2)

    for j in rules:
        for k in rules:
            if j!=k:
                fusedCooef = []
                for i in range(len(LE_cooef1)-1):
                    if (i == 0):

                        fusedCooef.append(fuseCoeff(LE_cooef1[0], LE_cooef2[0],j))
                    else:

                        c1 = fuseCoeff(LE_cooef1[i][0], LE_cooef2[i][0], k)
                        c2 = fuseCoeff(LE_cooef1[i][1], LE_cooef2[i][1], k)
                        c3 = fuseCoeff(LE_cooef1[i][2], LE_cooef2[i][2], k)

                        fusedCooef.append((c1,c2,c3))
                fusedImage = pywt.waverec2(fusedCooef, wavelet)

               
                fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)

                fusedImage = fusedImage.astype(np.uint8)
                name="fusedimages/"+ite+"/LE_"+j+"(LL)_"+k+"(rest).jpg"
                cv2.imwrite(name,fusedImage)
                #cv2.imshow("win",fusedImage)
                #cv2.waitKey(0)

