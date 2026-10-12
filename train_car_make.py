import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat


BASE = "data"                  
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 10
MAX_MAKES = 163               


names = loadmat(os.path.join(BASE, "misc", "/Users/rishikora/projects/projects/Computer Vision/Dataset/data/misc/make_model_name.mat"))["make_names"].ravel()

def read_list(filename):
    path = os.path.join(BASE, "train_test_split", "classification", filename)
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def make_id(rel_path):
    return int(rel_path.split("/")[0])      

def build_dataset(paths, class_ids, shuffle=False):
    files = [os.path.join(BASE, "image", p) for p in paths]
    labels = [class_ids[make_id(p)] for p in paths]
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img, label
    if shuffle:
        ds = ds.shuffle(len(files))
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_paths = read_list("train.txt")
test_paths = read_list("test.txt")
chosen_ids = sorted({make_id(p) for p in train_paths})[:MAX_MAKES]
class_ids = {mid: i for i, mid in enumerate(chosen_ids)}
class_names = [str(np.ravel(names[mid - 1])[0]) for mid in chosen_ids]
train_paths = [p for p in train_paths if make_id(p) in class_ids]
test_paths = [p for p in test_paths if make_id(p) in class_ids]
print("Makes:", class_names)
print("Train images:", len(train_paths), "Test images:", len(test_paths))

train_ds = build_dataset(train_paths, class_ids, shuffle=True)
test_ds = build_dataset(test_paths, class_ids)


base = keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                      include_top=False, weights="imagenet")
base.trainable = False
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(chosen_ids), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
loss, acc = model.evaluate(test_ds)
print("Test accuracy:", round(acc * 100, 2), "%")
model.save("compcars_make_model.keras")


def predict(image_path):
    img = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img)[None]
    probs = model.predict(arr, verbose=0)[0]
    for i in probs.argsort()[::-1][:3]:
        print(class_names[i], round(float(probs[i]) * 100, 1), "%")

predict(os.path.join(BASE, "image", test_paths[0]))