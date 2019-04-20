# Hindi Handwritten Digit Recognition
A handwritten digit recognition for devnagari characters in Hindi language written in Python. Basically an experimentation with different datasets available in the internet. Download the datasets from the following links:
dataset - 1 : [kaggle.com](https://www.kaggle.com/ashokpant/devanagari-character-dataset)
dataset - 2 : [archive.ics.uci.edu](http://archive.ics.uci.edu/ml/machine-learning-databases/00389/)


## Dataset -1
### Setup :
- Clone or download this repository
- Download the dataset, unzip it and place it in the `.dataset-1/data` folder.
- Go to the terminal, `cd ` into the project folder
- Make sure you have `python3` and `pip3` installed.
- Run `pip3 install -r requirements.txt` and it will install all project dependencies for you.

### Usage :
```
cd dataset-1

python3 clean.py

python3 train.py <argument>
USAGE : <argument> : -v - use vowel dataset
				   : -c - use consonant dataset
				   : -n - use numeral dataset (default)
python3 predict.py
```
