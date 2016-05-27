import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import transformations as tf
import numpy as np
import dataio
import cv2
import os


def plotcam(ax, r, t, col = [0.2, 0.2, 0.2], scale = 1.0):
    """
        Plot a camera with a given pose. Camera looks at z.
            \param ax : handle to the axes
            \param r: orientation
            \param t: position
            \param col: color of the camera
            \param scale: size of the camera
    """
    f = 7.0
    h = 3.0
    w = 4.0
    pp = np.array([[0, 0, 0], [w/2, -h/2, f], [w/2, h/2, f], [-w/2, h/2, f], [-w/2, -h/2, f]]).transpose()
    pp /= np.max(pp)
    pp *= scale
    pw = np.asarray(r).dot(pp) + np.asmatrix(t).transpose()
    pw = pw.transpose().tolist()

    for i in [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]:
        poly = Poly3DCollection([[pw[i[0]], pw[i[1]], pw[i[2]]]])
        poly.set_alpha(0.1)
        poly.set_color(col)
        ax.add_collection3d(poly)

    poly = Poly3DCollection([[pw[1], pw[2], pw[3], pw[4]]])
    poly.set_alpha(0.2)
    poly.set_color(col)
    ax.add_collection3d(poly)
    coloredaxes = np.asarray([pw[4], pw[1]])
    ax.plot(coloredaxes[:, 0], coloredaxes[:, 1], coloredaxes[:, 2], 'r', linewidth = 2.0)  # x
    coloredaxes = np.asarray([pw[4], pw[3]])
    ax.plot(coloredaxes[:, 0], coloredaxes[:, 1], coloredaxes[:, 2], 'g', linewidth = 2.0)  # y


def mergeimages(imagesList, numCols, resize = 1.0, nameOut = None):
    """
        Create image concatenating the input images.
            \param imagesList : list of paths to images
            \param numCols: number of images to concatenate horizontally (<0 makes one row)
            \param resize: resize factor of the final image
            \param nameOut: path to the image to save
            \preturn: the images concatenated
    """

    # Check if images have to be concatenated horizontally
    horizontal = 0
    if numCols < 0:
        numCols = len(imagesList)
        horizontal = 1

    g_imgRows = g_imgCols = g_imgChannels = 0;
    imgNum = 0
    for imageName in imagesList:
        try:
            img, _ = dataio.imread(imageName)
        except:
            print 'Image not valid: %s' % imageName
            continue
        if img is None:
            continue
        imgRows, imgCols, imgChannels = img.shape

        # Assign global values
        if imgNum == 0:
            imgNum += 1
            g_imgRows = imgRows
            g_imgCols = imgCols
            g_imgChannels = imgChannels
            imgMergeRow = img
            imgType = os.path.splitext(imageName)[1]
        else:
            # Discard images with different size
            if (imgRows != g_imgRows) | (imgCols != g_imgCols) | (imgChannels != g_imgChannels):
                print 'Skipped image: %s' % imageName
                continue

            imgNum = imgNum + 1
            if numCols == 1:
                imgMergeRow = np.concatenate((imgMergeRow, img))
                imgMerge = imgMergeRow
            elif imgNum % numCols == 0:
                imgMergeRow = np.concatenate((imgMergeRow, img), axis = 1)
                if imgNum == numCols:
                    imgMerge = imgMergeRow
                else:
                    imgMerge = np.concatenate((imgMerge, imgMergeRow))
            elif (imgNum - 1) % numCols == 0:
                imgMergeRow = img
            else:
                imgMergeRow = np.concatenate((imgMergeRow, img), axis = 1)

        print 'Added file: %s' % imageName

    # Check if last row is not full
    if imgMergeRow.shape[1] != numCols * g_imgCols:
        if horizontal == 0:
            imgMergeRow = np.concatenate((imgMergeRow, np.zeros(
                (g_imgRows, numCols * g_imgCols - imgMergeRow.shape[1], imgMergeRow.shape[2]), np.uint8)), axis = 1)
        if 'imgMerge' in locals():
            imgMerge = np.concatenate((imgMerge, imgMergeRow))
        else:
            imgMerge = imgMergeRow

    # Resize image
    if resize != 1:
        imgMerge = cv2.resize(imgMerge, (0, 0), fx = resize, fy = resize)

    # Save imageName
    if nameOut:
        cv2.imwrite(nameOut, imgMerge)
        print("Done: " + nameOut)
    return imgMerge


if __name__ == '__main__':

    # Test merge images
    if False:
        # imagesList = [f for f in os.listdir(imgsFolder) if
        #               os.path.isfile(os.path.join(imgsFolder, f))]  # and f[0] == 'd'
        # def _natural_sort_key(s):
        #     return [int(text) if text.isdigit() else text.lower() for text in re.split(re.compile('([0-9]+)'), s)]
        # imagesList.sort(key = _natural_sort_key)

        mergeimages(['../../samples/images/left.png', '../../samples/images/right.png'], 2, 0.2, '../../samples/images/merge.png')

    # Test plot camera
    if False:
        fig = plt.figure('camera')
        ax = Axes3D(fig)

        # plot camera 1
        r = tf.rotation_matrix(np.deg2rad(0), [0, 0, 1])[:3, :3]
        plotcam(ax, r, [0, 0, 0], scale = 0.5)

        # plot camera 2
        r = tf.rotation_matrix(np.deg2rad(90), [1, 0, 0])[:3, :3]
        plotcam(ax, r, [1, 0, 0], col = [0.2, 0.2, 0.2], scale = 0.5)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1))
        ax.autoscale_view()
        plt.show()