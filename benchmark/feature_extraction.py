import dlib
import cv2

from benchmark.util import is_grayscale, get_first_face_location, SettingNotSupportedError


def dlib_resnet(image, **kwargs):
    if is_grayscale(image):
        raise SettingNotSupportedError("Grayscale not supported for " + dlib_resnet.__name__)

    landmark_model = dlib.shape_predictor("models/face_cropping/shape_predictor_5_face_landmarks.dat")
    feature_model = dlib.face_recognition_model_v1("models/feature_extraction/dlib_face_recognition_resnet_model_v1.dat")

    face = get_first_face_location(image)
    shape = landmark_model(image, face)
    crop = dlib.get_face_chip(image, shape)

    def run():
        return feature_model.compute_face_descriptor(crop)

    return run


def openface(image, **kwargs):
    if is_grayscale(image):
        raise SettingNotSupportedError("Grayscale not supported for " + dlib_resnet.__name__)

    landmark_model = dlib.shape_predictor("models/face_cropping/shape_predictor_5_face_landmarks.dat")
    feature_model = cv2.dnn.readNetFromTorch("models/feature_extraction/nn4.small2.v1.t7")

    face = get_first_face_location(image)
    shape = landmark_model(image, face)
    crop = dlib.get_face_chip(image, shape)

    def run():
        crop_blob = cv2.dnn.blobFromImage(crop, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        feature_model.setInput(crop_blob)
        return feature_model.forward()[0]

    return run


METHODS = [dlib_resnet, openface]



