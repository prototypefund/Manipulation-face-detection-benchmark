# Runtime comparison of different face detection / recognition methods

## Why?

To figure out what can be reasonably run on NVIDIA Jetson. Not evaluating the quality
of the models, only runtime is relevant.

## Installation

Tested with Python 3.7. Before installing the requirements, check installation instructions
for dlib and opencv to make sure you have all necessary libraries installed.

```
pip install -r requirements.txt
sh download_models.sh
```

## Testing

There are no automatic tests. You can manually check that all models are running correctly,
you can run:

`
python test_manual.py
`

This will create a directory `results/test_images` where you can visually inspect
that the generated face detections and face croppings make sense.

## Running

```
python run_benchmarks.py
python plot.py
```
The generated metrics will be in `results/data`, generated plots in `results/plots`.