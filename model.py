import torch
import torchvision.models as models
import torch.nn as nn

class Img_Caption(nn.Module):
    def __init__(self, encoder, rnn, vocab_size, embed_dim,hidden_dim, num_rnn_layers = 1):
        super().__init__()
        self.encoder = encoder
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = rnn(embed_dim, hidden_dim, num_rnn_layers, batch_first= True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, img, caption, lengths):
        img_ftr = self.encoder(img)
        embed = self.embed(caption)
        
        embed = torch.cat((img_ftr.unsqueeze(1), embed), 1)
        decoder_input = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True) 
        hiddens, _ = self.rnn(decoder_input)
        out = self.output(hiddens[0])
        return out
        
def res50_encoder(num_out):
    res_model = models.resnet50(pretrained=True)
    for param in res_model.parameters():
        param.requires_grad = False
    num_ftrs = res_model.fc.in_features
    res_model.fc = nn.Linear(num_ftrs, num_out)
    return res_model