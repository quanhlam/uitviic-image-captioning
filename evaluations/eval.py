__author__ = 'tylin'
# import sys
import nltk
from ptbtokenizer import PTBTokenizer
from pyvi import ViTokenizer
from bleu import Bleu
from meteor import Meteor
from rouge import Rouge
from cider import Cider
# # from spice import Spice

TOKEN_BY_TOKEN = False
def tokenizer_sentence(data, tokenizer):
    if tokenizer == 'pyvi':
        for img_id, caption_data_list in data.items():
            for caption_data in caption_data_list:
                caption_data['caption'] = ViTokenizer.tokenize(caption_data['caption'].lower())
                if TOKEN_BY_TOKEN is True:
                    caption_data['caption'] = caption_data['caption'].replace('_', ' ')
    elif tokenizer == 'nltk':
        for img_id, caption_data_list in data.items():
            for caption_data in caption_data_list:
                caption_data['caption'] = ' '.join(nltk.tokenize.word_tokenize(caption_data['caption'].lower()))
    return data


class COCOEvalCap(object):
    def __init__(self, coco, cocoRes, tokenizer):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.tokenizer = tokenizer
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}    

        for imgId in imgIds:
            if imgId in self.coco.imgToAnns and imgId in self.cocoRes.imgToAnns:
                gts[imgId] = self.coco.imgToAnns[imgId]
                res[imgId] = self.cocoRes.imgToAnns[imgId]
        # =================================================
        # Set up scorers
        # =================================================
        print ('tokenization...')
        gts  = tokenizer_sentence(gts, self.tokenizer)
        res = tokenizer_sentence(res, self.tokenizer)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print ('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print ("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print ("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
