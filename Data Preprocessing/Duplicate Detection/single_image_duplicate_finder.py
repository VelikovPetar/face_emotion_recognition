# Finds duplicates of a single image by comparing the normalized rgb histograms with all images

import cv2
import os
import numpy as np

DIRECTORY = 'clean_dataset'                                                 # the directory with images
IMAGE = 'fc122b7f1091f33e3c7d1c4f3922f27019e4f975703da5a155033462.jpg'      # the single image being compared


# a function which takes 2 RGB histograms and calculates the distance between them
def hist_distance(hist1, hist2):
    # sizes (number of total pixels) of both images for normalization
    size1 = np.sum(hist1[0])
    size2 = np.sum(hist2[0])

    # all 6 individual histograms
    b_hist1 = hist1[0]
    g_hist1 = hist1[1]
    r_hist1 = hist1[2]

    b_hist2 = hist2[0]
    g_hist2 = hist2[1]
    r_hist2 = hist2[2]

    # normalize the histograms by dividing by the image size
    b_norm_hist1 = np.divide(b_hist1, size1)
    g_norm_hist1 = np.divide(g_hist1, size1)
    r_norm_hist1 = np.divide(r_hist1, size1)

    b_norm_hist2 = np.divide(b_hist2, size2)
    g_norm_hist2 = np.divide(g_hist2, size2)
    r_norm_hist2 = np.divide(r_hist2, size2)

    # return the sum of the absolute differences between the RGB histograms of both images
    return np.sum(np.absolute(np.subtract(b_norm_hist1, b_norm_hist2))) + np.sum(np.absolute(np.subtract(g_norm_hist1, g_norm_hist2))) + np.sum(np.absolute(np.subtract(r_norm_hist1, r_norm_hist2)))


matches = []

print('READING THE IMAGES')
cnt = 0
img1 = cv2.imread(DIRECTORY + '/' + IMAGE)
histogram1 = [cv2.calcHist([img1], [0], None, [256], [0, 256]), cv2.calcHist([img1], [1], None, [256], [0, 256]), cv2.calcHist([img1], [2], None, [256], [0, 256])]
for f in os.listdir(DIRECTORY):
    if f == IMAGE:
        continue
    cnt += 1
    if cnt % 1000 == 0:
        print(cnt)
    img2 = cv2.imread(DIRECTORY + '/' + f)
    histogram2 = [cv2.calcHist([img2], [0], None, [256], [0, 256]), cv2.calcHist([img2], [1], None, [256], [0, 256]), cv2.calcHist([img2], [2], None, [256], [0, 256])]
    distance = hist_distance(histogram1, histogram2)
    matches.append((f, distance))


# sort the matches by the distance
matches = sorted(matches, key=lambda tup: tup[1])


# show the matches with least distance
for m in matches:
    img = cv2.imread(DIRECTORY + '/' + m[0])

    print(m[0])
    cv2.imshow(m[0], img)
    cv2.waitKey()
    cv2.destroyAllWindows()
