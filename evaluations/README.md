## I. Introduction
Source code from [coco-caption](https://github.com/tylin/coco-caption); had modifications in terms of file directory and adding Vietnamese tokenizer for comparison purpose.

## II. Usage
- Move evaluation dataset to ```./data```
- Move machine output dataset to ```./results```
- Go to ```config.py``` to modify suitable filename
- Run ```python eval.py```  

The current eval.py is running BLEUs, CIDEr, ROUGE_L (line 59)


### Notes
- In file ```eval.py``` line 58, disable metric by commenting the code line if you don't want to use the metric.

## III. License
[coco-caption](https://github.com/tylin/coco-caption)
