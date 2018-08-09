# Facial features/landmarks - 68 (x, y) coordinates that map to facial structures of the face

# -Mouth
# -Right eyebrow
# -Left eyebrow
# -Right eye
# -Left eye
# -Nose
# -Jaw

import cv2
import dlib


class FaceNotDetectedException(Exception):
    pass


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
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


def draw_facial_features(image, features):
    for point in features:
        cv2.circle(image, point, 2, (0, 0, 255), -1)
    return image


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    _, frame = cam.read()
    try:
        features = detect_facial_features(frame, is_face_aligned=False)
        for point in features:
            cv2.circle(frame, point, 2, (0, 0, 255), -1)
        cv2.imshow("Output", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except FaceNotDetectedException:
        print("Face was not detected.")
