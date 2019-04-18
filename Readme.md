## Hindi Handwritten Digit Recognition
A handwritten digit recognition for Hindi language written in Python. Download the dataset from the following link [link](https://www.kaggle.com/ashokpant/devanagari-character-dataset)

### Usage
```
python3 clean.py

python3 train.py <argument>
USAGE : <argument> : v - use vowel dataset
				   : c - use consonant dataset
				   : n - use numeral dataset (default)
```
### How this works ?
- Download the dataset, unzip it and place it in the `data` folder.
- Go to the terminal, `cd ` into the project folder
- Make sure you have `python3` and `pip3` installed. Run `pip3 install -r requirements.txt` and it will install all project dependencies for you.
- Run `python3 clean.py`. This will scan the folders and images. Converts those images to numpy arrays and stores it in `data` folder.
- Run `python3 train.py`. This contains `train()` method which will import the `.npy` arrays with values and labels and train it, save the trained model in `data` folder. Then the `test()` method will import the trained model and predicts the values of arrays with the model.
