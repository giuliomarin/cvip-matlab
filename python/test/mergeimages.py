import cv2
import os
import numpy as np

joinFiles = ['/Users/giulio/Desktop/cvpr',
             '/Users/giulio/Desktop/eccv']
outDir = '/Users/giulio/Desktop/fused2'


def h2rgb(v1, v2, vh):
    if vh < 0:
        vh += 1
    if vh > 1:
        vh -= 1
    if (6 * vh) < 1:
        return v1 + (v2 - v1) * 6 * vh
    if (2 * vh) < 1:
        return v2
    if (3 * vh) < 2:
        return v1 + (v2 - v1) * ((2 / 3 - vh) * 6)
    return v1


def complementarycolor(r, g, b):
    cmin = min((r, g, b))
    cmax = max((r, g, b))
    cdelta = cmax - cmin
    l = (cmin + cmax) / 2
    if cdelta == 0:
        h = 0
        s = 0
    else:
        if l < 0.5:
            s = cdelta / (cmax + cmin)
        else:
            s = cdelta / (2 - cmax - cmin)

        deltar = (((cmax - r) / 6) + (cmax / 2)) / cmax
        deltag = (((cmax - g) / 6) + (cmax / 2)) / cmax
        deltab = (((cmax - b) / 6) + (cmax / 2)) / cmax

        if r == cmax:
            h = deltab - deltag
        elif g == cmax:
            h = (1 / 3) + deltar - deltab
        elif b == cmax:
            h = (2 / 3) + deltag - deltar

        if h < 0: h +=1
        if h > 1: h -=1

    # Compute opposite hue
    h2 = (h + 0.5) % 1

    # convert back to RGB
    if s == 0:
        ro = go = bo = l
    else:
        if l < 0.5:
            var2 = l * (1 + s)
        else:
            var2 = (l + s) - (s * l)
        var1 = 2 * l - var2
        ro = h2rgb(var1, var2, h2 + (1/3))
        go = h2rgb(var1, var2, h2)
        bo = h2rgb(var1, var2, h2 - (1/3))

    return ro, go, bo

def mergeimages(src1, src2, dst, text = None):
    m1 = cv2.imread(src1)
    m2 = cv2.imread(src2)
    m3 = np.abs(m1 - m2)

    # Add text
    if text:
        text.append('diff')
        # Parameters
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        thickness = 2
        thrinvert = 50
        defaultcolor = np.asarray([0, 0, 255])

        # m1
        textsize, baseline = cv2.getTextSize(text[0], fontface, fontscale, thickness)
        textorig = ((m1.shape[1] - textsize[0]) / 2, textsize[1] + thickness)
        colorimg = np.mean(m1[textorig[1]:textorig[1] + textsize[1], textorig[0]:textorig[0] + textsize[0], :], (0,1))
        print np.mean(np.abs(colorimg - defaultcolor))
        if (np.mean(np.abs(colorimg - defaultcolor))) > thrinvert:
            color = defaultcolor
        else:
            color = 255 - colorimg
        cv2.putText(m1, text[0], textorig, fontface, fontscale, color, thickness)

        # m2
        textsize, baseline = cv2.getTextSize(text[1], fontface, fontscale, thickness)
        textorig = ((m2.shape[1] - textsize[0]) / 2, textsize[1] + thickness)
        colorimg = np.mean(m2[textorig[1]:textorig[1] + textsize[1], textorig[0]:textorig[0] + textsize[0], :], (0, 1))
        if (np.mean(np.abs(colorimg - defaultcolor))) > thrinvert:
            color = defaultcolor
        else:
            color = 255 - colorimg
        cv2.putText(m2, text[1], textorig, fontface, fontscale, color, thickness)

        # m3
        textsize, baseline = cv2.getTextSize(text[2], fontface, fontscale, thickness)
        textorig = ((m3.shape[1] - textsize[0]) / 2, textsize[1] + thickness)
        colorimg = np.mean(m3[textorig[1]:textorig[1] + textsize[1], textorig[0]:textorig[0] + textsize[0], :], (0,1))
        if (np.mean(np.abs(colorimg - defaultcolor))) > thrinvert:
            color = defaultcolor
        else:
            color = 255 - colorimg
        cv2.putText(m3, text[2], textorig, fontface, fontscale, color, thickness)

    # Concatenate
    d = cv2.hconcat((m1, m2))
    d = cv2.hconcat((d, m3))
    cv2.imwrite(dst, d)
    return


if __name__ == "__main__":

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    for dirpath, dirsname, filesname in os.walk(joinFiles[0]):
        for filename in filesname:
            a = os.path.join(dirpath, filename)
            b = os.path.join(dirpath.replace(joinFiles[0], joinFiles[1]), filename)
            print 'Merging [%s] and [%s]' % (a, b)
            if not (os.path.exists(a) and os.path.exists(b)):
                continue
            outFile = os.path.join(outDir, os.path.basename(a))
            mergeimages(a, b, outFile, text = ['cvpr', 'eccv'])
