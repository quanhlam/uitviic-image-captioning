# Introduction
Baseline Image Captioning model from a pytorch tutorial. For more details go to [Pytorch tutorial](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning).  

## Usage

### 0. Configuration
Go to ``config.py`` to modify suitable parameters and filepath.

### 1. Preprocessing
```
python build_vocab.py   
python resize.py
```

### 2. Train the model
``` 
python train.py 
```

### 3. Evaluate the model
``` 
python outout.py
```

### 4. Test the model
``` 
python sample.py --image='<image file path>'
```
