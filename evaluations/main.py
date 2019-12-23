from coco import COCO
from eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab

import config


pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

coco = COCO(config.val_path)
cocoRes = coco.loadRes(config.res_file)

cocoEval = COCOEvalCap(coco, cocoRes, config.tokenizer)
# cocoEval.params['image_id'] = cocoRes.getImgIds()

cocoEval.evaluate()