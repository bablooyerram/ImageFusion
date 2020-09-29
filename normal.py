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


for ite in file.keys():

    fusionmethod = 'max'
    I1 = cv2.imread('Set1_10/'+ite+'/'+file[ite][0],0)
    I2 = cv2.imread('Set1_10/'+ite+'/'+file[ite][1],0)
    I1 = cv2.resize(I1,(512,512))
    I2 = cv2.resize(I2,(512,512)) 
 #   if(I2.shape!=I1.shape):
 #      if(ite=="Set-3(CT-PET Brain)" or ite=="Set-8(MRT1-T2 Brain)" or ite=="Set-9(MRT1-T2 Brain)"):
 #           I1 = cv2.resize(I1,I2.shape)
 #       else:
 #           I2 = cv2.resize(I2,I1.shape)
        
    wavelet = 'db1'
    cooef1 = pywt.wavedec2(I1[:,:], wavelet)
    cooef2 = pywt.wavedec2(I2[:,:], wavelet)
    
    fusedCooef = []
    print(ite)
    os.mkdir("fusedimages/"+ite)
    for j in rules:
        for k in rules:
            if j!=k:
                fusedCooef = []
                for i in range(len(cooef1)-1):
                    if (i == 0):

                        fusedCooef.append(fuseCoeff(cooef1[0], cooef2[0],j))
                    else:

                        c1 = fuseCoeff(cooef1[i][0], cooef2[i][0], k)
                        c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], k)
                        c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], k)

                        fusedCooef.append((c1,c2,c3))
                fusedImage = pywt.waverec2(fusedCooef, wavelet)


                fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)

                fusedImage = fusedImage.astype(np.uint8)
                
                name="fusedimages/"+ite+"/NOR_"+j+"(LL)_"+k+"(rest).jpg"
                cv2.imwrite(name,fusedImage)
                """cv2.imshow("win",fusedImage)
                cv2.waitKey(0)"""
