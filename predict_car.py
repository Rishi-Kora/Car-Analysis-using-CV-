import os
import sys
import numpy as np
from scipy.io import loadmat
from tensorflow import keras

BASE = os.path.join("Dataset", "data")
MODEL_FILE = os.path.join("Dataset", "compcars_make_model.keras")
IMG_SIZE = (224, 224)
MAX_MAKES = 163                      

names = loadmat(os.path.join(BASE, "misc", "make_model_name.mat"))["make_names"].ravel()
with open(os.path.join(BASE, "train_test_split", "classification", "train.txt")) as f:
    ids = sorted({int(line.split("/")[0]) for line in f if line.strip()})
class_names = [str(np.ravel(names[mid - 1])[0]) for mid in ids[:MAX_MAKES]]

photo = sys.argv[1]
model = keras.models.load_model(MODEL_FILE)
img = keras.utils.load_img(photo, target_size=IMG_SIZE)
arr = keras.utils.img_to_array(img)[None]
probs = model.predict(arr, verbose=0)[0]

print("Top 3 guesses for", photo)
for i in probs.argsort()[::-1][:3]:
    print(class_names[i], round(float(probs[i]) * 100, 1), "%")