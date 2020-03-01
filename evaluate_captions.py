from tqdm import tqdm, tqdm_notebook
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
from pycocotools.coco import COCO

def evaluate_captions( true_captions_path, generated_captions_path ):
    """
    Takes json formatted true and predicted captions, and calculates BLEU1, BLEU4 scores
    :param true_captions_path: path to json file with true COCO captions, used with pycocotools.coco
    :param generated_captions_path: path to json file with predicted COCO captions, used with pycocotools.coco
    
    :return: BLEU1, BLEU2 score tuple
    """
    
 
    
    score1 = 0
    score4 = 0

    smoother = SmoothingFunction()
    
    count = 0
    for candidate, reference in tqdm(zip(open(generated_captions_path), open(true_captions_path))) :
#         candidate = cocoRes.imgToAnns[i][0]['caption']
#         reference = []
#         for entry in coco.imgToAnns[i]:
#             reference.append(entry['caption'])
        count += 1
        score1 += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
        score4 += sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=smoother.method1)

    bleu1 = 100*score1/count
    bleu4 = 100*score4/count

        
    print("BLEU 1:", np.round(bleu1,2), 
          "BLEU 4:", np.round(bleu4,2))
    
    return bleu1, bleu4

## SAMPLE USAGE
"""
true_annotations_file ='./data/annotations/captions_val2014.json'
pred_annotations_file = 'baseline_lstm_captions.json'

BLEU1, BLEU4 = evaluate_captions( true_annotations_file, pred_annotations_file )
"""
