from PIL import Image,ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from scipy import ndimage,stats
from sklearn.mixture import GaussianMixture
import colorsys
import time
from skimage import filters,morphology
import os
'''
Esta parte contiene codigos de GMM
'''

def getGaussianMixturedImage(strImage):

    imArray1 = cv2.cvtColor(np.array(Image.open(strImage)),cv2.COLOR_BGR2GRAY)
    imArray2 = ndimage.gaussian_filter(np.array(imArray1), sigma=4)
    imArray3 = 0.5*imArray1 + 0.5*imArray2
    img = imArray3
    classif = GaussianMixture(n_components=2)
    classif.fit(img.reshape((img.size, 1)))
    threshold = np.mean(classif.means_)
    binary_img = img > threshold


    return (binary_img)



def getBackGround2(imageArray,center1,center2):


    rows = imageArray.shape[0]
    cols = imageArray.shape[1]
    vecCenter1 = []
    vecCenter2 = []
    dist1 = 0
    dist2 = 0

    i = 0
    j = 0
    rowStep = int(0.10*rows)
    colStep = int(0.10*cols)
    while (i < rows):
        while ( j < cols):
            pixel = imageArray[i,j]
            coordinates = []
            coordinates.append(i)
            coordinates.append(j)
            j = j + colStep
            if (pixel == center1):
                vecCenter1.append(coordinates.copy())
            else:
                vecCenter2.append(coordinates.copy())
        j = 0
        i = i + rowStep


    for i in range(len(vecCenter1)- 1):
        x = vecCenter1[i]
        y = vecCenter1[i+1]
        dist = math.sqrt(math.pow(x[0]-y[0],2)+math.pow(x[1]-y[1],2))
        dist1 = dist1 + dist

    for i in range(len(vecCenter2)- 1):
        x = vecCenter2[i]
        y = vecCenter2[i+1]
        dist = math.sqrt(math.pow(x[0]-y[0],2)+math.pow(x[1]-y[1],2))
        dist2 = dist2 + dist

    dist1 = dist1/len(vecCenter1)
    dist2 = dist2/len(vecCenter2)

    if (dist1 > dist2):
        backGround = center1
    else:
        backGround = center2

    return (backGround)



def getBackGround(imageArray,center1,center2):


    rows = imageArray.shape[0]
    cols = imageArray.shape[1]
    vecCenter1 = []
    vecCenter2 = []
    dist1 = 0
    dist2 = 0

    for k in range(200):
        i = np.random.randint(0,rows,1,int)
        j = np.random.randint(0,cols,1,int)
        pixel = imageArray[i,j]
        coordinates = []
        coordinates.append(i)
        coordinates.append(j)
        if (pixel == center1):
            vecCenter1.append(coordinates.copy())
        else:
            vecCenter2.append(coordinates.copy())

    for i in range(len(vecCenter1)- 1):
        x = vecCenter1[i]
        y = vecCenter1[i+1]
        dist = math.sqrt(math.pow(x[0]-y[0],2)+math.pow(x[1]-y[1],2))
        dist1 = dist1 + dist

    for i in range(len(vecCenter2)- 1):
        x = vecCenter2[i]
        y = vecCenter2[i+1]
        dist = math.sqrt(math.pow(x[0]-y[0],2)+math.pow(x[1]-y[1],2))
        dist2 = dist2 + dist

    dist1 = dist1/len(vecCenter1)
    dist2 = dist2/len(vecCenter2)

    if (dist1 > dist2):
        backGround = center1
    else:
        backGround = center2

    return (backGround)


def binaryToGrayScale(imageArray1,backGround,strImage):

    imageArray2 = np.array(Image.open(strImage).convert('L'))
    rows = imageArray1.shape[0]
    cols = imageArray1.shape[1]

    for i in range(rows):
        for j in range(cols):
            pixel = imageArray1[i,j]
            if (pixel == backGround):
                imageArray2[i,j] = 255
            else:
                imageArray2[i,j] = 0

    imageArray3 = ndimage.gaussian_filter(imageArray2,20)

    return (imageArray1,imageArray2)


def binaryToGrayScale2(imageArray1,backGround,strImage):

    imageArray2 = imageArray1.copy()
    rows = imageArray1.shape[0]
    cols = imageArray1.shape[1]

    for i in range(rows):
        for j in range(cols):
            pixel = imageArray1[i,j]
            if (pixel == backGround):
                imageArray2[i,j] = False
            else:
                imageArray2[i,j] = True


    return (imageArray2)


def analyzeImage(imageArray2,strImage):

    imageArray3 = np.array(Image.open(strImage))
    rows = imageArray2.shape[0]
    cols = imageArray2.shape[1]
    area = 0
    colorCount = np.zeros(5)
    colorID = ["white spots","nechrosis","chlorosis","healthy","unidentified"]
    index = 0

    for i in range(rows):
        for j in range(cols):
            pixel = imageArray2[i,j]
            if (pixel == True):
                area = area + 1
                rgb = imageArray3[i,j]
                index = getColor(rgb,1)
                colorCount[index] = colorCount[index] + 1



    finish1 = time.time()
    '''
    saveStr = "
    if ".jpg" in strImage:
        saveStr = strImage.replace(".jpg","res.png")
    if ".JPG" in strImage:
        saveStr = strImage.replace(".JPG","res.png")
    '''
    colorCount = colorCount * (1/area)

    plt.figure()
    plt.subplot(121)
    plt.title("P.B:  " + str(round(colorCount[0],2))
    + " - Necrosis: " + str(round(colorCount[1],2)) + "  - Clorosis: "
    + str(round(colorCount[3],2)) + " - Sano: " + str(round(colorCount[2],2)) )
    plt.imshow(imageArray3)
    plt.subplot(122)
    plt.title(" tiempo: segundos " + str(round(finish1-start1,4)) + " s")
    plt.imshow(imageArray2)
    plt.show()

    return(colorCount)




def codifyImage(binaryImage,background,rows,cols):

    binaryImage2 = binaryImage.copy()
    if background == True:
        for i in range(rows):
            for j in range(cols):
                pixel = binaryImage[i,j]
                if pixel == background:
                    binaryImage2[i,j] = False
                else:
                    binaryImage2[i,j] = True
    return binaryImage2.copy()



def evaluateSegmentation(binaryImage,strImage):

    imageArray2 = np.array(Image.open(strImage).convert('L'))
    rows = imageArray2.shape[0]
    cols = imageArray2.shape[1]
    pixels = []
    truePixels = []
    falsePixels = []

    for i in range(rows):
        for j in range(cols):
            pixels.append(imageArray2[i][j])

            if (binaryImage[i][j] == True):
                truePixels.append(imageArray2[i][j])

            else:
                falsePixels.append(imageArray2[i][j])

    variance1 = np.var(falsePixels)
    variance2 = np.var(pixels)
    f = variance1/variance2
    return(f)

'''
Esta parte contiene los codigos de saliency
'''



def maps(strimage):
    array1 = np.array(Image.open(strimage))
    rows = array1.shape[0]
    cols = array1.shape[1]
    array2 = np.zeros([rows, cols])
    pt = 0
    xc = 0
    yc = 0
    for i in range(rows):
        for j in range(cols):
            pixel = getColor(array1[i, j],2)
            array2[i,j] = pixel
            if pixel == 1:
                pt = pt + 1
                xc = xc + i
                yc = yc + j

    xc = xc/pt
    yc = yc/pt
    pt = pt/(rows*cols)
    s = np.zeros([rows,cols]) + 0.15
    sm = s.copy()
    side = int((0.65*max(rows,cols))/2)
    centers = []
    step = int(0.15*max(rows,cols))
    for i in range(3):
        for j in range(3):
            centers.append([max(int(xc-i*step),0),max(int(yc-j*step),0)])
            if j > 0:
                centers.append([min(int(xc+i*step),rows-1),min(int(yc+j*step),cols-1)])


    for k in range(15):

        cr,cc = centers[k]

        lr = max(cr-side, 0)
        hr = min(cr+side, rows-1)
        lc = max(cc-side, 0)
        hc = min(cc+side, cols-1)
        lrk = int(1.075*lr)
        hrk = int(0.925*hr)
        lck = int(1.075*lc)
        hck = int(0.925*hc)

        tk = 0
        tw = 0
        pk = 0
        pw = 0

        for i in range(lr, hr):
            for j in range(lc, hc):
                color = array2[i, j]
                if lrk < i < hrk and lck < j < hck:
                    if color == 1:
                        pk = pk + 1
                    tk = tk + 1

                else:
                    if color == 1:
                        pw = pw + 1
                    tw = tw + 1

        pk = pk/tk
        pw = pw/tw

        for i in range(lrk, hrk):
            for j in range(lck, hck):
                color = array2[i, j]
                if color == 1:
                    cs = s[i, j]
                    ccs = 1-cs
                    pkcs = pk*cs
                    ns = 0.8*pkcs/(pkcs + pw*ccs) + 0.2*pkcs/(pkcs + pt*ccs)
                    s[i,j] = ns
                    if ns > sm[i,j]:
                        sm[i,j] = ns


    tre = filters.threshold_otsu(sm)
    binary = sm > tre
    opening = morphology.opening(binary)
    #Para los casos en los que se tiene una hoja o planta sin espacios enete si.
    #closing = morphology.closing(opening,morphology.square(int(0.04*max(rows,cols))))
    #Para los casos en los que se tiene una planta con espacios y no una hoja.
    closing = morphology.closing(opening)

    area = 0
    colorCount =  np.zeros(5)
    for i in range(rows):
        for j in range(cols):
            if (closing[i,j] == True):
                area = area + 1
                rgb= array1[i,j]
                color = getColor(rgb,1)
                colorCount[color] = colorCount[color] + 1

    colorCount = colorCount * (1/area)

    finish2 = time.time()
    '''
    if ".jpg" in strImage:
        saveStr = strImage.replace(".jpg","res.png")
    if ".JPG" in strImage:
        saveStr = strImage.replace(".JPG","res.png")

    '''

    plt.figure()
    plt.subplot(121)
    plt.title("P.B:  " + str(round(colorCount[0],2))
    + " - Necrosis: " + str(round(colorCount[1],2)) + "  - Clorosis: "
    + str(round(colorCount[3],2)) + " - Sano: " + str(round(colorCount[2],2)) )
    plt.imshow(array1)
    plt.subplot(122)
    plt.title(" tiempo: segundos " +  str(round(finish2-start1,4))  + " s")
    plt.imshow(closing)
    plt.show()
    return (colorCount)


def getColor(rgb,id):

    r = rgb[0]/255
    g = rgb[1]/255
    b = rgb[2]/255

    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    if ( s <= 0.10 and v >= 0.9):
        if id == 1:
            color = 0
        else:
            color = 0
    elif(v <= 0.05):
        if id == 1:
            color = 1
        else:
            color = 0
    elif(h <= 45/360 and s > 0.15):
        if id == 1:
            color = 1
        else:
            color = 0
    elif(45/360 < h and h <= 55/360 and s > 0.15 and v < 0.5):
        if id == 1:
            color = 1
        else:
            color = 0
    elif(45/360 < h and h <= 55/360 and s > 0.3 and v >= 0.5):
        if id == 1:
            color = 3
        else:
            color = 1

    elif(55/360 < h and h <= 65/360 and s > 0.3):
        if id == 1:
            color = 3
        else:
            color = 1
    elif(65/360 < h  and h <= 175/360 and s >= 0.15):
        if id == 1:
            color = 2
        else:
            color = 1

    else:
        if id == 1:
            color = 4
        else:
            color = 0

    return color



#Parte nueva para evaluar desempeño de la segmentacion


def getGroundTruth(strImage):
    imageArray3 = np.array(Image.open(strImage))
    rows = imageArray3.shape[0]
    cols = imageArray3.shape[1]
    arrayGT = np.full((rows,cols),True)
    im = Image.new(mode="1", size=(cols,rows),color= 0)
    a1,a2,a3,a4 = imageArray3[rows-1][cols-1]


    #Para 15b,19b,20b

    '''
    a1,a2,a3 = imageArray3[rows-1][cols-1]
    for i in range(rows):
        for j in range(cols):
            r,g,b = imageArray3[i][j]
            if ( r > 250 and g > 250 and b > 250):
                arrayGT[i][j] = False
                im.putpixel((j,i),1)

    '''

    for i in range(rows):
        for j in range(cols):
            r,g,b,d = imageArray3[i][j]
            if ( r == a1 and g == a2 and b == a3):
                arrayGT[i][j] = False
                im.putpixel((j,i),1)


    return (arrayGT,im)

def calculatePrformance(gt,seg):
    rows = gt.shape[0]
    cols = gt.shape[1]
    trueArea = 0
    overEstimate = 0
    underEstimate = 0
    for i in range(rows):
        for j in range(cols):
            if(gt[i][j] == True):
                trueArea = trueArea + 1
            if(gt[i][j] == True and seg[i][j] == False):
                underEstimate = underEstimate + 1
            if (gt[i][j] == False and seg[i][j] == True):
                overEstimate = overEstimate + 1

    underEstimateP = (trueArea - underEstimate)/ trueArea
    overEstimateP = (overEstimate/trueArea)

    return(underEstimateP,overEstimateP)


import os
cwd = os.getcwd()
imagesString = []
for subdir, dirs, files in os.walk(cwd, topdown=True):
    del dirs[:]  # remove the sub directories.
    for file in files:
        if '.jpg' in file or '.JPG' in file:
            imagesString.append(str(file))

f = open("resultados4.txt", "w")

start = time.time()

for strImage in imagesString:

    print(strImage)
    start1 = time.time()
    imageArray3 = np.array(Image.open(strImage))
    rows = imageArray3.shape[0]
    cols = imageArray3.shape[1]
    binaryImage = (getGaussianMixturedImage(strImage))
    backGround = getBackGround2(binaryImage,True,False)
    binaryImage2 = codifyImage(binaryImage,backGround,rows,cols)
    #Para los casos en los que se tiene una hoja o planta sin espacios entre si.
    #opening = morphology.opening(binaryImage)
    closing = morphology.closing(binaryImage2,morphology.square(int(0.04*max(rows,cols))))
    #Para los casos en los que se tiene una planta con espacios y no una hoja.
    #closing = morphology.closing(binaryImage)
    #backGround = getBackGround(closing,True,False)
    f1 = evaluateSegmentation(closing,strImage)


    if f1 < 0.016:
        diagnosis = (analyzeImage(closing,strImage))
        f.write(str(round(diagnosis[0],3)) + "," + str(round(diagnosis[1],3)) + "," + str(round(diagnosis[2],3)) + ","  + str(round(diagnosis[3],3)) + "," + str(round(diagnosis[4],3)) + ","  + str(round(f1,3)) + "\n")

    else:
        diagnosis = maps(strImage)
        f.write(str(round(diagnosis[0],3)) + "," + str(round(diagnosis[1],3)) + "," + str(round(diagnosis[2],3)) + "," + str(round(diagnosis[3],3)) + "," + str(round(diagnosis[4],3)) + ","  + str(round(f1,3)) + "\n")



    '''
    diagnosis = (analyzeImage(closing,strImage))
    f.write(str(round(diagnosis[0],3)) + "," + str(round(diagnosis[1],3)) + "," + str(round(diagnosis[2],3)) + ","  + str(round(diagnosis[3],3)) + "," + str(round(diagnosis[4],3)) + ","  + str(round(f1,3)) + "\n")
    '''


f.close()


finish = time.time()
print(finish-start)
