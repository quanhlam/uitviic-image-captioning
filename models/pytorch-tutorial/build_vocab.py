from collections import Counter

import nltk
import pickle
from pyvi import ViTokenizer, ViPosTagger

from pycocotools.coco import COCO

from config import Config

class Vocabulary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = {}
		self.idx = 0
	
	def __call__(self, word):
		if not word in self.word2idx:
			return self.word2idx['<unk>']
		return self.word2idx[word]
	
	def __len__(self):
		return len(self.word2idx)

	def add_word(self, word):
		if not word in self.word2idx:
			self.word2idx[word] = self.idx
			self.idx2word[self.idx] = word
			self.idx +=1

def build_vocab(json, threshold, tokenizer):
	coco = COCO(json)
	counter = Counter()
	ids = coco.anns.keys()
	for i, id in enumerate(ids):
		caption = str(coco.anns[id]['caption'])
		tokens = tokenize(caption.lower(), tokenizer)
		counter.update(tokens)

		if (i+1) % 1000 == 0:
			print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

	# If the word frequency is less than 'threshold', then the word is discarded.
	words = [word for word, cnt in counter.items() if cnt >= threshold]

	# Create a vocab wrapper and add some special tokens.
	vocab = Vocabulary()
	vocab.add_word('<pad>')
	vocab.add_word('<start>')
	vocab.add_word('<end>')
	vocab.add_word('<unk>')

	# Add the words to the vocabulary.
	for i, word in enumerate(words):
		vocab.add_word(word)
	return vocab

def tokenize(caption, tokenizer):
	if tokenizer == 'pyvi':
		return ViTokenizer.tokenize(caption.lower()).split()
	if tokenizer == 'nltk':
		return nltk.tokenize.word_tokenize(caption.lower())

def main():
	config = Config()
	vocab = build_vocab(json=config.train_path, threshold=config.threshold, tokenizer = config.tokenizer)
	print(vocab.idx2word)
	vocab_path = config.vocab_path
	with open(vocab_path, 'wb') as f:
		pickle.dump(vocab, f)

	print(f"Vocabulary size: {len(vocab)}")
	print(f"Vocabluary wrapper path: {vocab_path}")

if __name__ == "__main__":
	main()