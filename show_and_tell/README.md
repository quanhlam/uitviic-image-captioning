## Introduction
Show and tell (NICv1) is state-of-the-art Image Captioning model for MS-COCO English captions in 2015. For more information about running the code, go to [Show and Tell](https://github.com/nikhilmaram/Show_and_Tell).

Note that now there is a upgraded version of Show and tell (NICv2), published in 2017.

## Installation
Download pretrained CNN model [here](https://ucsb.box.com/s/pj4gg3vpei57cf9xewttoqn01qqa3uj4).
Put the pretrained model in ```./models/vgg16_weights.npz```

## Usage

### 0. Configuration
Go to ``config.py`` to modify suitable parameters and filepath.

### 1. Train the model
``` 
python main.py --phase=train \
    --load_cnn \
    --cnn_model_file='./vgg16_weights.npz'\
    [--train_cnn]
```

### 2. Evaluate the model
``` 
python main.py --phase=eval \
    --model_file='./models/xxxxxx.npy'
```
