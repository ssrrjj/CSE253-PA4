from pycocotools.coco import COCO
import nltk
from tqdm import tqdm

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add(self, word):
        if word in self.word2idx:
            return
        self.word2idx[word] = self.idx
        self.idx2word[self.idx] = word
        self.idx += 1

    def __call__(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        return self.word2idx['<unk>']

    def __getitem__(self, item):
        assert (item < self.idx)
        return self.idx2word[item]


def build_vocab(json):
    coco = COCO(json)
    vocab = Vocabulary()
    vocab.add('<unk>')
    vocab.add('<start>')
    vocab.add('<end>')
    for key in tqdm(coco.anns.keys()):
        cap = coco.anns[key]['caption']
        tokens = nltk.tokenize.word_tokenize(str(cap).lower())
        for token in tokens:
            vocab.add(token)
    return vocab

