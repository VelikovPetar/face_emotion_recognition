# Face detection and aligning (rotation, scaling and cropping) of images
# TUTORIAL AND CODE: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import os

cnt = 0

INPUT_DIRECTORY = '/home/david/Desktop/cleaned dataset/images'
OUTPUT_DIRECTORY = '/home/david/Desktop/crop images'
SHAPE_PREDICTOR = 'shape_predictor_68_face_landmarks.dat'
DESIRED_WIDTH = 128

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
fa = FaceAligner(predictor, desiredFaceWidth=DESIRED_WIDTH)

for f in os.listdir(INPUT_DIRECTORY):
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(INPUT_DIRECTORY + '/' + f)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 2)

    # ignore images with more faces
    if len(faces) > 1:
        continue

    # loop over the face detections
    for face in faces:
        # extract the ROI of the *original* face, then align the face using facial landmarks
        (x, y, w, h) = rect_to_bb(face)

        # in case of negative values
        x = max(x, 0)
        y = max(y, 0)

        # print((x, y, w, h))
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=DESIRED_WIDTH)
        faceAligned = fa.align(image, gray, face)

        # show the original input image and detect faces in the grayscale image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("Input", image)

        # display the output images
        cv2.imshow("Original", faceOrig)
        cv2.imshow("Aligned", faceAligned)
        cv2.waitKey(0)

        # save the result
        cv2.imwrite(OUTPUT_DIRECTORY + '/' + f, faceAligned)

    # print progress
    cnt += 1
    print(cnt)