## Dataset - 2

### Setup :
- Clone or download this repository
- Download the [dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/00389/), unzip it and place it in the `./d2-conv/data` folder.
- Open terminal, `cd` into the project folder

### Usage :
```
docker build -t tf-python-ml
docker run -it -v $(pwd):/usr/src/app tf-python-ml bash
python clean.py
python train.py
python predict.py
```