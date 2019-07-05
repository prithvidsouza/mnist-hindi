## Dataset -1

### Setup :
- Clone or download this repository
- Download the [dataset](https://www.kaggle.com/ashokpant/devanagari-character-dataset), unzip it and place it in the `./d1-nn/data` folder.
- Open terminal, `cd` into the project folder

### Usage :
```
docker build -t python/tensorflow .
docker run --rm -it -v $(pwd):/usr/src/app python/tensorflow bash
python clean.py
python train.py
python predict.py
```