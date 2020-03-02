from data_loader import *
from evaluate_captions import *
import csv
from build_vocab import *
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from model import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json

train_json = './data/annotations/captions_train2014.json'
test_json = './data/annotations/captions_val2014.json'
train_root = './data/images/train/'
test_root = './data/images/test/'
vocab = build_vocab(train_json)
with open('TrainImageIds.csv', 'r') as f:
    reader = csv.reader(f)
    trainIds = list(reader)
trainIds = [int(i) for i in trainIds[0]]
#train_dataset = CocoDataset(train_root, train_json, trainIds, vocab)

valIds = trainIds[-len(trainIds)//5:]
trainIds = trainIds[:-len(trainIds)//5]

with open('TestImageIds.csv', 'r') as f:
    reader = csv.reader(f)
    testIds = list(reader)
testIds = [int(i) for i in testIds[0]]
#test_dataset = CocoDataset(test_root, test_json, testIds, vocab)

batch_size = 256
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

tsfm = transforms.Compose([
        transforms.Resize(size=(300,300)),
        transforms.ToTensor(),
        normalize,
    ])
train_loader = get_loader(train_root, train_json, trainIds, vocab, 
                          transform=tsfm, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          mode = 'train',
                          num_workers=4)
val_loader = get_loader(train_root, train_json, valIds, vocab, 
                          transform=tsfm, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          mode = 'train',
                          num_workers=4)
test_loader = get_loader(test_root, test_json, testIds, vocab, 
                          transform=tsfm, 
                          batch_size=batch_size, 
                          shuffle=False, 
                          mode = 'test',
                          num_workers=4)

#pretrained word embedding
# !curl  -L http://nlp.stanford.edu/data/glove.6B.zip>glove.6B.zip
# !unzip glove.6B.zip
pretrain_embed = False
if pretrain_embed:
    word2vec = {}
    idx = 0
    with open("glove.6B.300d.txt","rb") as f:
        for l in tqdm(f.readlines()):
            line = l.decode().split()
            word2vec[line[0]] = [float(x) for x in line[1:]]
            idx+=1
    pretrained_weight = np.zeros((vocab.idx, 300))
    for i in range(vocab.idx):
        word = vocab[i]
        if word in word2vec:
            pretrained_weight[i] = word2vec[word]
        else:
            pretrained_weight[i] = np.random.randn(300)

embed_dim = 300
vocab_size= vocab.idx
hiddem_dim = 512


baseline = Img_Caption(encoder= res50_encoder(embed_dim), rnn=nn.RNN, 
                        vocab_size=vocab_size, 
                        embed_dim=embed_dim,
                        hidden_dim=hiddem_dim,
                        num_rnn_layers = 2,
                        embed_weight = torch.tensor(pretrained_weight) if pretrain_embed else None
                      )

optimizer = optim.Adam(baseline.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


use_gpu = torch.cuda.is_available()
if use_gpu:
    baseline = baseline.cuda()


def train(mod, epochs):
    mod.train()
    best_loss = float('inf')
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        losss = []
        ts = time.time()
        for i, (imgs, caps, lengths) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                imgs = imgs.cuda()# Move your inputs onto the gpu
                caps = caps.cuda()# Move your labels onto the gpu
                #lengths = lengths.cuda()
            
            outputs = mod(imgs, caps, lengths)
            targets = nn.utils.rnn.pack_padded_sequence(caps, lengths, batch_first=True)[0]
            loss = criterion(outputs, targets)
            losss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if i % 500 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, i, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        # torch.save(fcn_model, 'best_model')
        
        train_loss.append(np.mean(losss))
        epoch_loss = val(mod)
        val_loss.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(mod, 'best_model')
        mod.train() 
    return train_loss,val_loss
def val(mod):
    mod.eval()
    
    ts = time.time()
    val_loss = 0
    for i, (imgs, caps, lengths) in enumerate(val_loader):
        optimizer.zero_grad()
        
        if use_gpu:
            imgs = imgs.cuda()# Move your inputs onto the gpu
            caps = caps.cuda()# Move your labels onto the gpu
            #lengths = lengths.cuda()

        outputs = mod(imgs, caps, lengths)
        targets = nn.utils.rnn.pack_padded_sequence(caps, lengths, batch_first=True)[0]
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        
        if i % 100 == 0:
            print("iter{}, loss: {}".format(i, loss.item()))
    val_loss/=len(val_loader)
    print('validation loss:', val_loss)
    print("Finish validation time elapsed {}".format(time.time() - ts))
    return val_loss

def ppl(mod):
    mod.eval()
    test_loss = 0
    for i, (imgs, caps, lengths) in enumerate(test_loader):
        optimizer.zero_grad()
        
        if use_gpu:
            imgs = imgs.cuda()# Move your inputs onto the gpu
            caps = caps.cuda()# Move your labels onto the gpu
            #lengths = lengths.cuda()

        outputs = mod(imgs, caps, lengths)
        targets = nn.utils.rnn.pack_padded_sequence(caps, lengths, batch_first=True)[0]
        loss = criterion(outputs, targets)
        test_loss += loss.item()
    test_loss/=len(test_loader)
    print('Test loss:', test_loss)
    ppl = np.exp(test_loss)
    print('Perplexity:', ppl)
    return test_loss, ppl

epochs  = 10
train_loss1, val_loss1 = train(baseline, epochs)


def idxtowords(idxs):
    words = []
    for idx in idxs:
        if idx == 3:
            break
        words.append(vocab[idx])
    return " ".join(words[1:])
def test(mod):
    mod.eval()
    if os.path.exists('caption.txt'):
        os.remove('caption.txt')
    if os.path.exists('generation.txt'):
        os.remove('generation.txt')
    capfile = open('caption.txt', 'a')
    genfile = open('generation.txt', 'a')
    anns = []
    idx = 0
    for i, (imgs, caps, lengths) in enumerate(test_loader):
    
        if use_gpu:
            imgs = imgs.cuda()# Move your inputs onto the gpu
            caps = caps.cuda()# Move your labels onto the gpu
            #lengths = lengths.cuda()
            
        feature = mod.encoder(imgs)
        sampled_ids = mod.sample(feature,max_length = 20)
        
         
#         sampled_ids = sampled_ids[0].cpu().numpy()
        
#         sampled_caption = []
        
#         for word_id in sampled_ids:
#             word = vocab.idx2word[word_id]
#             sampled_caption.append(word)
#             if word == '<end>':
#                 break
                
#         sentence = ' '.join(sampled_caption)
#         print (sentence)
        
#         sampled_caption = []
        
#         for word_id in caps[0].cpu().numpy():
#             word = vocab.idx2word[word_id]
#             sampled_caption.append(word)
#             if word == '<end>':
#                 break
#         sentence = ' '.join(sampled_caption)
#         print(sentence)
#         plt.imshow(np.asarray(imgs[0].cpu().permute(1,2,0)))
        sampled_ids = sampled_ids.cpu().numpy()
        caps = caps.cpu().numpy()
        for i in range(len(caps)):
            capfile.write(idxtowords(caps[i])+'\n')
            genfile.write(idxtowords(sampled_ids[i])+'\n')
            ann = {'image_id':testIds[idx], 'caption':idxtowords(sampled_ids[i])}
            anns.append(ann)
            idx+=1
    with open('res.json','w') as fp:
        
        json.dump(anns, fp)
        
#         imgs = np.asarray(imgs.cpu().permute(0,2,3,1))
#         for i in range(10):
#             generate = idxtowords(sampled_ids[i])
#             label = idxtowords(caps[i])
#             print('label:',label)
#             print('generate:',generate)
#             img = imgs[i]
#             img-=np.min(img)
#             img/=np.max(img)
#             plt.imshow(img)
#             plt.show()
#         break
        
def test_stochastic(mod,temp):
    mod.eval()
    if os.path.exists('caption.txt'):
        os.remove('caption.txt')
    if os.path.exists('generation.txt'):
        os.remove('generation.txt')
    capfile = open('caption.txt', 'a')
    genfile = open('generation.txt', 'a')
    generate_caps = []
    anns = []
    idx = 0
    for i, (imgs, caps, lengths) in enumerate(test_loader):
    
        if use_gpu:
            imgs = imgs.cuda()# Move your inputs onto the gpu
            caps = caps.cuda()# Move your labels onto the gpu
            #lengths = lengths.cuda()
            
        feature = mod.encoder(imgs)
        sampled_ids = mod.Stochastic_sample(feature,max_length = 20,temp = temp)
        
        sampled_ids = sampled_ids.cpu().numpy()
        caps = caps.cpu().numpy()
        for i in range(len(caps)):
            capfile.write(idxtowords(caps[i])+'\n')
            genfile.write(idxtowords(sampled_ids[i])+'\n')
            ann = {'image_id':testIds[idx], 'caption':idxtowords(sampled_ids[i])}
            
            anns.append(ann)
            idx+=1
    with open('res.json','w') as fp:
        
        json.dump(anns, fp)
    capfile.close()
    genfile.close()


baseline = torch.load("best_model")
ppl(baseline)

#test_stochastic(mod,0.1)
test(baseline)

BLEU_1 , BLEU_4  = evaluate_captions('./data/annotations/captions_val2014.json', 'res.json')