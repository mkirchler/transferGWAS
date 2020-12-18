# Multi-task training on EyePACS dataset

This directory contains all the code for the EyePACs pretraining task. We provide the trained models (use `../download_models.sh`), but you can use this code to retrain the model. You can also use the code to train on your own data, but since different imaging data sets usually require different preprocessing steps and hyperparameters, we would recommend using these scripts only for retinal fundus images with continuous target variable.

## Data preprocessing


### Images
This training script works on the EyePACs dataset which can be accessed at https://www.kaggle.com/c/diabetic-retinopathy-detection/data You will need a kaggle account, setup kaggle on your machine (follow https://github.com/Kaggle/kaggle-api) and you'll need to accept the competition rules.
Download all files (e.g. via the `kaggle` API), merge the `.zip` files and unzip those. Then you can use the `resize.py` script to crop and resize the images for faster loading. Resizing will likely take a few hours. All together:
```bash
kaggle competitions download -c diabetic-retinopathy-detection
unzip diabetic-retinopathy-detection.zip
rm diabetic-retinopathy-detection.zip sampleSubmission.csv.zip sample.zip

cat train.zip.* > train.zip
unzip train.zip

cat test.zip.* > train.zip
unzip test.zip

rm train.zip train.zip.* test.zip test.zip.*

python resize.py train/ test/ data/ --size 672

rm -r train/ test/
```

### Labels
Training-set labels will be downloaded together with the image data, but test-set labels have to be downloaded separately. To train on the full dataset you need to concatenate the two label files, e.g. via:
```bash
unzip trainLabels.csv.zip
rm trainLabels.csv.zip

# get the test labels; in case this link breaks in the future, check: https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/16149
wget https://storage.googleapis.com/kaggle-forum-message-attachments/90528/2877/retinopathy_solution.csv

python -c "import pandas as pd; pd.concat([pd.read_csv('trainLabels.csv'), pd.read_csv('retinopathy_solution.csv').drop('Usage', 1)]).to_csv('full_labels.csv', index=False)"
```

## Training

Once the data is preprocessed, you can train the network via:

```bash
python training_main.py data/ full_labels.csv \
            --epochs 100 \
            --res_depth 50 \    # ResNet depth; 18, 34, or 50
            --size 448 \        # input image size
            --bs 25 \           # batch size
            --lr 1e-3 \         # learning rate
            --num_workers 1 \
            --dev cuda:0 \
            --save_path models/eyepacs_pretrained.pt
```

(in case you used other variables in the preprocessing steps you will need to replace `data/` with your image directory and `full_labels.csv` with your label file).
See `python training_main.py -h` for additional arguments.
Per default, this will save the best final model into the `models/` directory.

### Weights & Biases tracking

If you want to monitor training progress through `Weights & Biases`, just set the `WANDB = True` flag at the beginning of the `training_main.py` file. You will first need to setup an account at https://app.wandb.ai/ and configure your credentials.

## Training on another dataset

In order to train on a different dataset, you will need to move all images into a shared directory (we call it `IMG_DIR`) in `.jpeg` format  and provide the labels in a `.csv` file with columns `image` and `level`. The `image` column needs to contain the filename per image (without extension), and the `level` contains the target; e.g.:
```
image,level
1_left,0.4
1_right,2.1
2_left,-0.1
...
```

## Using the trained embeddings

The training script will only save the body of the network (you can easily modify the code to also save the different heads in case you need it). To use later, you can simply load the weights again in pytorch via:

```python
import torch
from torchvision import models

model = models.resnet50() # assuming you trained with res_depth = 50
model.load_state_dict(torch.load(PATH_TO_YOUR_MODEL, map_location='cpu'))
```

To extract the features for GWA analysis, use the `feature_condensation` code.
