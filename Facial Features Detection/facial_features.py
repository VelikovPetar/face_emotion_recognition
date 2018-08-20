# Facial features/landmarks - 68 (x, y) coordinates that map to facial structures of the face

"""
Sets of points representing the each facial feature.

-Mouth: [48, 67]
-Left eye: [42, 47]
 Right eye: [36, 41]
-Nose: [27, 35]
-Left eyebrow: [22, 26]
-Right eyebrow: [17, 21]
-Jaw: [0, 16]
"""

import cv2
import dlib
import math


class FaceNotDetectedException(Exception):
    pass


class FacialFeatures:
    JAW = "jaw"
    RIGHT_EYEBROW = "right_eyebrow"
    LEFT_EYEBROW = "left_eyebrow"
    NOSE = "nose"
    RIGHT_EYE = "right_eye"
    LEFT_EYE = "left_eye"
    MOUTH = "mouth"

    FACIAL_FEATURES_IDXS = {
        JAW: (0, 16),
        RIGHT_EYEBROW: (17, 21),
        LEFT_EYEBROW: (22, 26),
        NOSE: (27, 35),
        RIGHT_EYE: (36, 41),
        LEFT_EYE: (42, 47),
        MOUTH: (48, 67)
    }


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def detect_facial_features(image, is_face_aligned=False):
    """
    Calculates the coordinates of the 68 facial features/landmarks.

    :param image: image of the face to detect the facial features
    :param is_face_aligned: indicator whether the face is aligned. If False, face detection will be performed
    :return: list of tuples (x, y) representing the coordinates of the facial features
    :raises FaceNotDetectedException if the face detection fails
    """
    greyscale = image
    h, w = greyscale.shape

    if is_face_aligned:
        rect = dlib.rectangle(left=0, top=0, right=w, bottom=h)
    else:
        rects = detector(greyscale, 1)
        if len(rects) == 0:
            raise FaceNotDetectedException()
        rect = rects[0]

    shape = predictor(greyscale, rect)
    features = []
    for i in range(0, 68):
        features.append((shape.part(i).x, shape.part(i).y))
    return features


def euclidean_distance(point1, point2):
    """
    Calculates euclidean distance between 2 points.

    :param point1: the first point
    :param point2: the second point
    :return: the euclidean distance
    """
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def distances_between_facial_features(features):
    """
    Calculates the distances between each point of facial features.

    :param features: the list of facial features
    :return: list containing the distances between each pair of facial features
    """
    distances = []
    for (i, point1) in enumerate(features):
        for (j, point2) in enumerate(features):
            if j > i:
                distances.append(euclidean_distance(point1, point2))
    return distances


def draw_facial_feature(image, features, feature_name):
    """
    Draws a single facial feature on the image.

    :param image: the image to draw facial feature
    :param features: the list of detected facial feature
    :param feature_name: the identifier of the facial feature
    :return: image with drawn facial feature
    """
    if feature_name not in FacialFeatures.FACIAL_FEATURES_IDXS.keys():
        raise Exception("Invalid facial feature.")
    feature_points = FacialFeatures.FACIAL_FEATURES_IDXS.get(feature_name)
    start = feature_points[0]
    end = feature_points[1] + 1
    for i in range(start, end):
        cv2.circle(image, features[i], 2, (0, 0, 255), -1)
    return image


def draw_all_facial_features(image, features):
    for point in features:
        cv2.circle(image, point, 2, (0, 0, 255), -1)
    return image


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    _, frame = cam.read()
    try:
        features = detect_facial_features(frame, is_face_aligned=False)
        frame = draw_facial_feature(frame, features, FacialFeatures.LEFT_EYE)
        cv2.imshow("Output", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(distances_between_facial_features(features))
    except FaceNotDetectedException:
        print("Face was not detected.")
