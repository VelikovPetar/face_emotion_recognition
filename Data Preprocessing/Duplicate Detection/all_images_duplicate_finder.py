# Finds duplicates by comparing the normalized rgb histograms between all pairs of images

import cv2
import os
import numpy as np

THRESHOLD = 0.7                 # the difference threshold for detecting duplicates, usually 0.5 to 0.7 works
DIRECTORY = 'dataset'           # the directory with images


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


# crate a {filename: [b_hist, g_hist, r_hist]} dict of all images
histograms = {}
for f in os.listdir(DIRECTORY):
    img = cv2.imread(DIRECTORY + '/' + f)
    histograms[f] = [cv2.calcHist([img], [0], None, [256], [0, 256]), cv2.calcHist([img], [1], None, [256], [0, 256]), cv2.calcHist([img], [2], None, [256], [0, 256])]


# brute-force comparison of all pairs of images
cnt = 0
duplicates = []
images = histograms.keys()
for f1 in images:
    print(cnt)      # print progress
    cnt += 1
    for f2 in images:
        if f1 <= f2:    # prevent processing same pair of images twice
            continue
        score = hist_distance(histograms[f1], histograms[f2])   # calculate distance and compare with threshold
        if score < THRESHOLD:
            duplicates.append((f1, f2, score))


# sort the duplicates by distance and print the results
duplicates = sorted(duplicates, key=lambda tup: tup[2])
for m in duplicates:
    print(m[2])
    print(m[0])
    print(m[1])
    print()


# show the duplicates visually
for m in duplicates:
    filename1 = DIRECTORY + '/' + m[0]
    filename2 = DIRECTORY + '/' + m[1]
    if os.path.isfile(filename1) and os.path.isfile(filename2):
        img1 = cv2.imread(filename1)
        img2 = cv2.imread(filename2)

        cv2.imshow(filename1, img1)
        cv2.imshow(filename2, img2)
        cv2.waitKey()
        cv2.destroyAllWindows()
