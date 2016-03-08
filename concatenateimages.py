import os
import sys
import cv2
import numpy as np
import re

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)] 

## Merge images in folder imgsFolder
def mergeimages(imgsFolder, numCols, resize = 1, nameOut = None):
    imagesList = [f for f in os.listdir(imgsFolder) if os.path.isfile(os.path.join(imgsFolder, f))] # and f[0] == 'd'
    
    imagesList.sort(key = natural_sort_key)
    
    # Check if images have to be concatenated horizontally
    horizontal = 0
    if numCols < 0:
        numCols = len(imagesList)
        horizontal = 1
    
    g_imgRows = g_imgCols = g_imgChannels = 0;
    imgNum = 0
    for imageName in imagesList:
        img = cv2.imread(os.path.join(imgsFolder, imageName))
        if img is None:
            continue
        imgRows, imgCols, imgChannels = img.shape
        
        # Assign global values
        if imgNum == 0:
            imgNum = imgNum + 1
            g_imgRows = imgRows
            g_imgCols = imgCols
            g_imgChannels = imgChannels
            imgMergeRow = img
            imgType = os.path.splitext(os.path.join(imgsFolder, imageName))[1]
        else:
            # Discard images with different size
            if (imgRows != g_imgRows) | (imgCols != g_imgCols) | (imgChannels != g_imgChannels):
                print 'Skipped image: %s' % imageName
                continue
                
            imgNum = imgNum + 1
            if numCols == 1:
                imgMergeRow = cv2.vconcat((imgMergeRow, img))
                imgMerge = imgMergeRow
            elif imgNum % numCols == 0:
                imgMergeRow = cv2.hconcat((imgMergeRow, img))
                if imgNum == numCols:
                    imgMerge = imgMergeRow
                else:
                    imgMerge = cv2.vconcat((imgMerge, imgMergeRow))
            elif (imgNum - 1) % numCols == 0:
                imgMergeRow = img
            else:
                imgMergeRow = cv2.hconcat((imgMergeRow, img))
        
        print 'Added file: %s' % imageName
    
    # Check if last row is not full
    if imgMergeRow.shape[1] != numCols * g_imgCols:
        if horizontal == 0:
            imgMergeRow = cv2.hconcat((imgMergeRow, np.zeros((g_imgRows, numCols * g_imgCols - imgMergeRow.shape[1] , imgMergeRow.shape[2]), np.uint8)))
        if 'imgMerge' in locals():
            imgMerge = cv2.vconcat((imgMerge, imgMergeRow))
        else:
            imgMerge = imgMergeRow
            
    # Resize image
    if resize != 1:
        imgMerge = cv2.resize(imgMerge, (0,0), fx = resize, fy = resize)
            
    # Save imageName
    if nameOut:
        if nameOut[0] == '.':
            imgOutName = os.path.join(imgsFolder, nameOut + imgType)
        else:
            imgOutName = nameOut
    else:
        imgOutName = os.path.join(imgsFolder, 'merged' + imgType)
    cv2.imwrite(imgOutName, imgMerge)
    print("Done: " + imgOutName)
    return imgMerge


if __name__ == "__main__":
    
    # Default values
    numCols = 5 # -1 for horizontal concatenation
    imgsFolder = '/Users/giulio/Desktop/ordered'
    #os.path.dirname(os.path.realpath(__file__))
    resize = 1
    
    # Read input values
    if len(sys.argv) > 1:
        if len(sys.argv) != 3:
            print 'Image concatenation (row major)\nusage: python %s imagesFolder numCols' % sys.argv[0]
            sys.exit()
        imgsFolder = sys.argv[1]
        numCols = int(sys.argv[2])
    else:
        imgsFolder = raw_input(('Folder with the images: '))
        numCols = int(raw_input(('Number of columns: ')))
        resize = raw_input(('Resize [1]: '))
        if len(resize) > 0:
            resize = float(resize)
        else:
            resize = 1
    
    mergeimages(imgsFolder, numCols, resize, './merge')