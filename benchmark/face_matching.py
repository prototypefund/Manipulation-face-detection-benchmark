import numpy as np


def numpy_euclidean(face, face_database):
    def run():
        return [np.linalg.norm(face - saved_face) for saved_face in face_database]

    return run


#def opencv_euclidean(face, face_database):
#    def run():
#        pass


METHODS = [numpy_euclidean]
