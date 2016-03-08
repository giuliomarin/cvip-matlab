import cv2
import os
import shutil
import sys
sys.path.append('/GitHub/cvip/')
import mergeimages as mi

joinFiles = [r'/Data/2_NitroImages/highres_simulation/raw/3_jb/DisparityRGB/disparityRGB_%d.png',
             r'/Data/2_NitroImages/highres_simulation/raw/3_ms/DisparityRGB/disparityRGB_%d.png']
outDir = r'/Data/2_NitroImages/highres_simulation/21_30_fixed'

def mergeImages(src1, src2, dst):
    m1 = cv2.imread(src1);
    m2 = cv2.imread(src2);
    d = cv2.hconcat((m1, m2));

    cv2.imwrite(dst, d);
    return;

if not os.path.exists(outDir):
    os.makedirs(outDir);

for id in range(1,11):
    a, b = joinFiles[0] % id, joinFiles[1] % id
    print id
    if not (os.path.exists(a) and os.path.exists(b)):
        continue
    outFile = os.path.join(outDir, os.path.basename(a)) # 'masterDisp_%d.png' % id
    mergeImages(a, b, outFile)

# mi.mergeimages(outDir, 1, 0.5, '/Data/NitroImages/bestAllNoSame.png')