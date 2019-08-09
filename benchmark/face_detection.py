import dlib
import cv2

from benchmark.util import SettingNotSupportedError, ExperimentFailure, is_grayscale

"""
All detection methods have to return a list of detected bounding boxes with format [left, right, top, bottom]
"""


def dlib_hog(image, **kwargs):
    detector = dlib.get_frontal_face_detector()

    def convert_rect(rect):
        return [rect.left(), rect.right(), rect.top(), rect.bottom()]

    def run():
        result = list([convert_rect(r) for r in detector(image, 0)])
        if (len(result)) == 0:
            raise ExperimentFailure("No faces found by " + dlib_hog.__name__)
        return result

    return run


def dlib_cnn(image, **kwargs):
    detector = dlib.cnn_face_detection_model_v1("models/face_detection/mmod_human_face_detector.dat")

    def convert_rect(rect):
        return [rect.rect.left(), rect.rect.right(), rect.rect.top(), rect.rect.bottom()]

    def run():
        result = list([convert_rect(r) for r in detector(image, 0)])
        if (len(result)) == 0:
            raise ExperimentFailure("No faces found by " + dlib_hog.__name__)
        return result

    return run


def opencv_haar(image, **kwargs):
    detector = cv2.CascadeClassifier('models/face_detection/haarcascade_frontalface_default.xml')

    # not part of benchmark because would supply image in correct color order in main code
    if not is_grayscale(image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def convert_rect(rect):
        x, y, w, h = list(rect)
        return [x, x+w, y+h, y]

    def run():
        result = list([convert_rect(r) for r in detector.detectMultiScale(image)])
        if (len(result)) == 0:
            raise ExperimentFailure("No faces found by " + dlib_hog.__name__)
        return result

    return run


def opencv_cnn(image, **kwargs):
    if is_grayscale(image):
        raise SettingNotSupportedError("Grayscale not supported for " + opencv_cnn.__name__)

    model_file = "models/face_detection/opencv_face_detector_uint8.pb"
    config_file = "models/face_detection/opencv_face_detector.pbtxt"
    detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)

    scale_x = image.shape[1] / 300
    scale_y = image.shape[0] / 300

    def convert_rect(rect):
        left, bottom, right, top = rect[3:7] * 300
        return [int(left*scale_x), int(right*scale_x), int(top*scale_y), int(bottom*scale_y)]

    def run():
        # part of benchmark because this is extra work that we  would not do when not using this model
        image_resized = cv2.resize(image, (300, 300))
        image_scaled = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), (104.0, 177.0, 123.0))

        detector.setInput(image_scaled)
        result = list(detector.forward()[0,0])
        if (len(result)) == 0:
            raise ExperimentFailure("No faces found by " + dlib_hog.__name__)

        return [convert_rect(r) for r in result]

    return run


METHODS = [dlib_hog, dlib_cnn, opencv_haar, opencv_cnn]
