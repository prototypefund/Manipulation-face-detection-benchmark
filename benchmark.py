import timeit
from itertools import product
import json
from pathlib import Path

import dlib
import cv2


class SettingNotSupportedError(Exception):
    pass


def load_image(image_path, scaleFactor=None, convert_to_grayscale=False):
    image = cv2.imread(image_path)
    assert image.shape == (720, 1280, 3)
    if scaleFactor:
        image = cv2.resize(image, (0, 0), fx=scaleFactor, fy=scaleFactor)

    if convert_to_grayscale:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def dlib_hog(image):
    detector = dlib.get_frontal_face_detector()

    def run():
        list(detector(image, 0))

    return run


def dlib_cnn(image):
    detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

    def run():
        list(detector(image, 0))

    return run


def opencv_haar(image):
    detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    if len(image.shape) == 3:  # is not grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def run():
        list(detector.detectMultiScale(image))

    return run


def opencv_cnn(image):
    model_file = "models/opencv_face_detector_uint8.pb"
    config_file = "models/opencv_face_detector.pbtxt"
    detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)

    if len(image.shape) == 2:
        raise SettingNotSupportedError("Grayscale not supported")

    def run():
        image_resized = cv2.resize(image, (300, 300))
        image_scaled = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), (104.0, 177.0, 123.0))
        detector.setInput(image_scaled)
        list(detector.forward())

    return run


IMAGE_PATH = "images/test.png"
RESULT_FILE = "results/runtimes.json"
REPLICATIONS = 10

assert Path(IMAGE_PATH).exists()

scales = [1.0, 0.75, 0.25]
grayscale = [False, True]
methods = [dlib_hog, dlib_cnn, opencv_haar, opencv_cnn]

results = []
for scale, gray, method in product(scales, grayscale, methods):
    image_data = load_image(IMAGE_PATH, scale, gray)
    try:
        runtime = timeit.Timer(method(image_data)).repeat(REPLICATIONS, 1)
        results.append({"scale": scale, "grayscale": gray, "method": method.__name__, "runtime": runtime})
    except SettingNotSupportedError:
        pass

Path(RESULT_FILE).parent.mkdir(parents=True, exist_ok=True)
Path(RESULT_FILE).write_text(json.dumps(results, indent=2))
