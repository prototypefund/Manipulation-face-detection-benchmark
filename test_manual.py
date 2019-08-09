import cv2
import copy

from run_benchmarks import *


IMAGE_DIRECTORY = Path("images")
RESULTS_DIRECTORY = Path("results/test_images")


def get_filename(setting):
    setting = copy.copy(setting)
    setting["method"] = setting["method"].__name__
    return "_".join(["{}={}".format(name, setting[name]) for name in sorted(setting.keys())])


def draw_face_detections(setting, result, filename):
    image = load_image(IMAGE_DIRECTORY / setting["image"], setting["scale"], setting["grayscale"])

    for rect in result:
        cv2.rectangle(image, (rect[0], rect[2]), (rect[1], rect[3]), (0, 0, 255), 2)

    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_face_cropping(setting, result, filename):
    cv2.imwrite(filename, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def check_feature_extractions(setting, result, filename):
    assert result.shape[0] == 128


def check_face_matching(setting, result, filename):
    assert len(result) == setting["num_registered_persons"]


def run(benchmark, result_handler):
    output_dir = RESULTS_DIRECTORY / benchmark.__class__.__name__
    output_dir.mkdir(parents=True, exist_ok=True)

    for setting in dict_product(benchmark.params):
        print(setting)
        method = setting["method"]
        data = benchmark.setup(setting)

        try:
            result = method(**data)()
        except SettingNotSupportedError as e:
            print(e)
            continue

        outfile = str((output_dir / get_filename(setting))) + ".png"
        result_handler(setting, result, outfile)


run(FaceDetectionScaleBenchmark(), draw_face_detections)
run(FaceDetectionPersonsBenchmark(), draw_face_detections)
run(FaceCroppingBenchmark(), save_face_cropping)
run(FeatureExtractionBenchmark(), check_feature_extractions)
run(FaceMatchingBenchmark(), check_face_matching)
