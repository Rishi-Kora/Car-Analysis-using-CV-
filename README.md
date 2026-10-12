# Car-Analysis-using-CV (Computer Vision)

A beginner-friendly computer vision project. It trains a model to look at a car photo and guess the car make (for example, Audi, Toyota, or BMW).

> **Licence:** The CompCars dataset is for **non-commercial research only**. Do not use the trained model or the dataset in a paid or commercial product. Read the official CompCars terms before using it.

## What it does

- Trains an image classifier on car photos from the CompCars dataset.
- Uses a pre-trained **MobileNetV2** model (from ImageNet) and adds a new last layer that predicts the car make.
- Saves the trained model to `compcars_make_model.keras`.
- Predicts the top 3 most likely car makes for any photo you choose.

## Results

| Setting | Makes | Epochs | Test accuracy |
|---|---|---|---|
| Frozen base model | 163 | 5 | 4.08% (first try, before shuffling) |
| Frozen base model, shuffled data | 163 | 5 | 39.46% |
| Frozen base model, shuffled data | 163 | 10 | 42.41% |

Random guessing would be right less than 1% of the time for 163 makes.

## Folder layout

```
Computer Vision/
├── train_car_make.py
├── predict_car.py
├── my_car.jpg                  (your own test photo)
└── Dataset/
    ├── compcars_make_model.keras   (created after training)
    └── data/
        ├── image/
        ├── label/
        ├── misc/
        ├── part/
        └── train_test_split/
```

## Requirements

- Python 3.10 to 3.12
- On a Mac with Apple Silicon, install the Mac version of TensorFlow:

```
pip install tensorflow-macos tensorflow-metal
pip install scipy pillow
```

On other computers:

```
pip install tensorflow scipy pillow
```

## How to use

### 1. Get the dataset

Download CompCars from the official site (see the Dataset section below). Extract it so that the `data` folder is inside the `Dataset` folder.

### 2. Train the model

Run from inside the `Dataset` folder:

```
cd Dataset
python ../train_car_make.py
```

Training takes about 10 to 12 minutes for 10 epochs on a MacBook Pro. At the end it prints the test accuracy and saves `compcars_make_model.keras`.

### 3. Predict the car make for a photo

Run from the `Computer Vision` folder, with the photo in the same folder:

```
cd "Computer Vision"
python predict_car.py my_car.jpg
```

Example output:

```
Top 3 guesses for my_car.jpg
Audi 80.7 %
Baojun 2.8 %
MAZDA 2.6 %
```

## Settings you can change

In `train_car_make.py`:

| Setting | What it does | Notes |
|---|---|---|
| `MAX_MAKES` | How many car makes to learn | Use a smaller number (for example, 10) for a quick test |
| `EPOCHS` | How many training rounds | More rounds take longer and may not improve results |
| `BATCH` | Photos studied at once | Lower it if the computer runs out of memory |
| `base.trainable` | Whether the pre-trained layers can change | `False` gave the best result so far. `True` gave a much lower result with the current settings |

`MAX_MAKES` in `predict_car.py` must match the value used in training.

## Dataset

- **Name:** CompCars (Comprehensive Cars)
- **Source:** http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html
- **Citation:** Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. "A Large-Scale Car Dataset for Fine-Grained Categorization and Verification", CVPR 2015.
- Some files are password-protected. Get the password and instructions from the official site.
- Training uses the official lists `train_test_split/classification/train.txt` and `test.txt`.
- The car make is the first number in each photo's path (for example, `78/1/2010/xyz.jpg` means make `78`). Make names are in `misc/make_model_name.mat`.

## Known limitations

- The model learns from car photos only. It may guess wrongly for photos of other objects, parts, or unusual angles.
- Some makes have few photos, so the model is less accurate for them.
- Test accuracy is about 42%, which is useful for learning but not good enough for a production product.
- Line 15 of `train_car_make.py` contains a full path to the author's Mac. Change it to a relative path (`os.path.join(BASE, "misc", "make_model_name.mat")`) before sharing the code.

## Next steps

- Add more training data or use a stronger pre-trained model.
- Try fine-tuning with a lower learning rate.
- Use the CompCars car-type attribute for a second model.
