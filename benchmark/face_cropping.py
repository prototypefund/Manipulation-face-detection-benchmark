import dlib

from benchmark.util import is_grayscale, get_first_face_location, SettingNotSupportedError


def dlib_5point(image, **kwargs):
    if is_grayscale(image):
        raise SettingNotSupportedError("Grayscale not supported for " + dlib_5point.__name__)

    landmark_model = dlib.shape_predictor("models/face_cropping/shape_predictor_5_face_landmarks.dat")

    face = get_first_face_location(image)

    def run():
        shape = landmark_model(image, face)
        return dlib.get_face_chip(image, shape)

    return run


def dlib_68point(image, **kwargs):
    if is_grayscale(image):
        raise SettingNotSupportedError("Grayscale not supported for " + dlib_5point.__name__)

    landmark_model = dlib.shape_predictor("models/face_cropping/shape_predictor_68_face_landmarks.dat")

    face = get_first_face_location(image)

    def run():
        shape = landmark_model(image, face)
        return dlib.get_face_chip(image, shape)

    return run


def simple_crop(image, **kwargs):
    face = get_first_face_location(image)

    def run():
        return image[face.top():face.bottom(), face.left():face.right()]

    return run


METHODS = [dlib_5point, dlib_68point, simple_crop]

