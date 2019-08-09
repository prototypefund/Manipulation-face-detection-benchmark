import timeit
import json
from pathlib import Path

import numpy as np

from benchmark import face_detection, face_cropping, feature_extraction, face_matching
from benchmark.util import load_image, SettingNotSupportedError, dict_product


IMAGE_DIRECTORY = Path("images")
RESULTS_DIRECTORY = Path("results/data")
REPLICATIONS = 1


class FaceDetectionScaleBenchmark:
    params = {"image": ["1_large.jpg"],
              "scale": [1.0, 0.75, 0.5, 0.25],
              "grayscale": [True, False],
              "method": face_detection.METHODS}

    @staticmethod
    def setup(setting):
        return {"image": load_image(IMAGE_DIRECTORY / setting["image"], setting["scale"], setting["grayscale"])}


class FaceDetectionPersonsBenchmark:
    params = {"image": ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg"],
              "scale": [1.0],
              "grayscale": [True, False],
              "method": face_detection.METHODS}

    @staticmethod
    def setup(setting):
        return {"image": load_image(IMAGE_DIRECTORY / setting["image"], setting["scale"], setting["grayscale"])}


class FaceCroppingBenchmark:
    params = {"image": ["1_large.jpg"],
              "scale": [1.0, 0.75, 0.5, 0.25],
              "grayscale": [True, False],
              "method": face_cropping.METHODS}

    @staticmethod
    def setup(setting):
        return {"image": load_image(IMAGE_DIRECTORY / setting["image"], setting["scale"], setting["grayscale"])}


class FeatureExtractionBenchmark:
    params = {"image": ["1_large.jpg"],
              "scale": [1.0, 0.75, 0.5, 0.25],
              "grayscale": [False, True],
              "method": feature_extraction.METHODS}

    @staticmethod
    def setup(setting):
        return {"image": load_image(IMAGE_DIRECTORY / setting["image"], setting["scale"], setting["grayscale"])}


class FaceMatchingBenchmark:
    params = {"num_registered_persons": [1, 10, 100, 1000],
              "method": face_matching.METHODS}

    @staticmethod
    def setup(setting):
        face = np.random.rand(128)
        face_database = np.random.rand(setting["num_registered_persons"], 128)
        return {"face": face, "face_database": face_database}


def run_benchmark(benchmark, replications: int):
    results = []
    
    for setting in dict_product(benchmark.params):
        print(setting)

        method = setting["method"]

        try:
            data = benchmark.setup(setting)
            setting["runtime"] = timeit.Timer(method(**data)).repeat(replications, 1)
            setting["method"] = setting["method"].__name__
            results.append(setting)
        except SettingNotSupportedError as e:
            print(e)

    result_file = RESULTS_DIRECTORY / (benchmark.__class__.__name__ + ".json")
    result_file.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    run_benchmark(FaceDetectionScaleBenchmark(), REPLICATIONS)
    run_benchmark(FaceDetectionPersonsBenchmark(), REPLICATIONS)
    run_benchmark(FaceCroppingBenchmark(), REPLICATIONS)
    run_benchmark(FeatureExtractionBenchmark(), REPLICATIONS)
    run_benchmark(FaceMatchingBenchmark(), REPLICATIONS)
