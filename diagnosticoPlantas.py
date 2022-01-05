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
                index = getColorNN2(rgb)
                colorCount[index] = colorCount[index] + 1

    colorCount = colorCount * (1/area)
    end = time.time()
    print(end-start)

    return(colorID,colorCount)


def getColorNN2(rgb):

    color = 0

    r = rgb[0]/255
    g = rgb[1]/255
    b = rgb[2]/255

    h,s,v = colorsys.rgb_to_hsv(r,g,b)

    o11 = 1/(1 + math.exp(-1*(-3.1580227 - 2.1237742*h + 0.7001064*s + 2.8366065*v)))
    o12 = 1/(1 + math.exp(-1*(-52.0145977 + 234.8668015*h + 0.6990048*s + 30.0214358*v)))
    o13 = 1/(1 + math.exp(-1*(-1.253056 + 13.992049*h + 43.986046*s - 18.551682*v)))
    o14 = 1/(1 + math.exp(-1*(-4.905778 + 59.807990*h - 4.249384*s - 10.403728*v)))
    o21 = 1/(1 + math.exp(-1*(0.6325434 - 1.2561720*o11 - 3.0853635*o12 + 2.9881184*o13 - 79.1063440*o14)))
    o22 = 1/(1 + math.exp(-1*(-0.3035949 - 0.3958577*o11 + 0.7328440*o12 - 5.9774242*o13 - 1.4438876*o14)))
    o23 = 1/(1 + math.exp(-1*(-19.876105 + 358.356456*o11 + 8.919032*o12 - 4.203202*o13 - 3.558226*o14)))

    output = round(1.003647 + 2.079147*o21 - 8.061846*o22 + 1.005889*o23,0)


    if (output <= 0):
        color = 0
    elif(output == 1):
        color = 1
    elif(output == 2):
        color = 3
    elif(output == 3):
        color = 2
    elif(output == 4):
        color = 1
    else:
        color = 4

    return (color)


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
            pixel = getColorNN2a(array1[i, j])
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
                color = getColorNNa(rgb)
                colorCount[color] = colorCount[color] + 1

    colorCount = colorCount * (1/area)

    finish = time.time()
    print(colorCount)
    print(finish-start)
    return (closing)


def getColorNN2a(rgb):

    r = rgb[0]/255
    g = rgb[1]/255
    b = rgb[2]/255

    h,s,v = colorsys.rgb_to_hsv(r,g,b)

    o11 = 1/(1 + math.exp(-1*(-3.1580227 - 2.1237742*h
    + 0.7001064*s + 2.8366065*v)))
    o12 = 1/(1 + math.exp(-1*(-52.0145977 + 234.8668015*h
    + 0.6990048*s + 30.0214358*v)))
    o13 = 1/(1 + math.exp(-1*(-1.253056 + 13.992049*h
    + 43.986046*s - 18.551682*v)))
    o14 = 1/(1 + math.exp(-1*(-4.905778 + 59.807990*h
    - 4.249384*s - 10.403728*v)))
    o21 = 1/(1 + math.exp(-1*(0.6325434 - 1.2561720*o11
    - 3.0853635*o12 + 2.9881184*o13 - 79.1063440*o14)))
    o22 = 1/(1 + math.exp(-1*(-0.3035949 - 0.3958577*o11
    + 0.7328440*o12 - 5.9774242*o13 - 1.4438876*o14)))
    o23 = 1/(1 + math.exp(-1*(-19.876105 + 358.356456*o11
    + 8.919032*o12 - 4.203202*o13 - 3.558226*o14)))

    output = (1.003647 + 2.079147*o21 - 8.061846*o22 + 1.005889*o23)
    #Usualmente se usa  0.15 pero generaba falsos positivos, se observo
    #que 0.04 genera mejores resultados.

    color = 0
    if abs(output - round(output,0)) < 0.1:
        output = round(output,0)
        if output <= 0:
            color = 0
        elif output == 1:
            color = 0
        elif output == 2:
            if g > r and g > b:
                color = 1
        elif output == 3:
            color = 1
        elif output == 4:
            color = 0
        else:
            color = 0
    else:
        color = 0

    return color



def getColorNNa(rgb):

    color = 0

    r = rgb[0]/255
    g = rgb[1]/255
    b = rgb[2]/255

    h,s,v = colorsys.rgb_to_hsv(r,g,b)

    o11 = 1/(1 + math.exp(-1*(-3.1580227 - 2.1237742*h
    + 0.7001064*s + 2.8366065*v)))
    o12 = 1/(1 + math.exp(-1*(-52.0145977 + 234.8668015*h
    + 0.6990048*s + 30.0214358*v)))
    o13 = 1/(1 + math.exp(-1*(-1.253056 + 13.992049*h
    + 43.986046*s - 18.551682*v)))
    o14 = 1/(1 + math.exp(-1*(-4.905778 + 59.807990*h
    - 4.249384*s - 10.403728*v)))
    o21 = 1/(1 + math.exp(-1*(0.6325434 - 1.2561720*o11
    - 3.0853635*o12 + 2.9881184*o13 - 79.1063440*o14)))
    o22 = 1/(1 + math.exp(-1*(-0.3035949 - 0.3958577*o11
    + 0.7328440*o12 - 5.9774242*o13 - 1.4438876*o14)))
    o23 = 1/(1 + math.exp(-1*(-19.876105 + 358.356456*o11
    + 8.919032*o12 - 4.203202*o13 - 3.558226*o14)))

    output = round(1.003647 + 2.079147*o21 - 8.061846*o22 + 1.005889*o23,0)


    if (output <= 0):
        color = 0
    elif(output == 1):
        color = 1
    elif(output == 2):
        color = 2
    elif(output == 3):
        color = 3
    elif(output == 4):
        color = 1
    else:
        color = 4

    return (color)


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



for l in range(1,24):
    strImage = "imagen" + str(l) + ".jpg"
    print(strImage)
    start = time.time()
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
    f1= evaluateSegmentation(closing,strImage)
    if f1 < 0.016:
        print(analyzeImage(closing,strImage))
        #strImage2 = "imagen" + str(l) + "b.jpg"
        #arrayGt,im = getGroundTruth(strImage2)
        #under,over = calculatePrformance(arrayGt,closing)
    else:
        closing = maps(strImage)
        #strImage2 = "imagen" + str(l) + "b.jpg"
        #arrayGt,im = getGroundTruth(strImage2)
        #under,over = calculatePrformance(arrayGt,closing)

