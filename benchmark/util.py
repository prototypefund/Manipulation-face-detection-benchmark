from itertools import product
from pathlib import Path

import cv2
import dlib


class SettingNotSupportedError(Exception):
    pass


class ExperimentFailure(Exception):
    pass


def load_image(image_path: Path, scaleFactor=None, convert_to_grayscale=False):
    image = cv2.imread(str(image_path))

    if scaleFactor:
        image = cv2.resize(image, (0, 0), fx=scaleFactor, fy=scaleFactor)

    if convert_to_grayscale:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def is_grayscale(image):
    return len(image.shape) == 2


# like itertools.product, but takes dictionary of lists and produces list of dictionaries containing all combinations
def dict_product(dict_of_lists):
    names = sorted(dict_of_lists.keys())
    values = [dict_of_lists[name] for name in names]

    for setting in product(*values):
        setting_dict = dict(zip(names, setting))
        yield setting_dict


def get_first_face_location(image):
    detector = dlib.get_frontal_face_detector()
    face_locations = detector(image, 0)
    if len(face_locations) == 0:
        raise SettingNotSupportedError("No faces found")
    if len(face_locations) > 1:
        print("More than one face found, using only the first one")
    return face_locations[0]
