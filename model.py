import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.elu(out)
        return out

class DrugVQA(torch.nn.Module):
    def __init__(self,args,block):
        super(DrugVQA,self).__init__()
        self.batch_size = args['batch_size']
        self.lstm_hid_dim = args['lstm_hid_dim']
        self.r = args['r']
        self.type = args['task_type']
        self.in_channels = args['in_channels']
        
        # RNN components
        self.embeddings = nn.Embedding(args['n_chars_smi'], args['emb_dim'])
        self.seq_embed = nn.Embedding(args['n_chars_seq'], args['emb_dim'])
        self.lstm = torch.nn.LSTM(args['emb_dim'], self.lstm_hid_dim, 2, batch_first=True, bidirectional=True, dropout=args['dropout'])
        self.linear_first = torch.nn.Linear(2*self.lstm_hid_dim, args['d_a'])
        self.linear_second = torch.nn.Linear(args['d_a'], args['r'])
        self.linear_first_seq = torch.nn.Linear(32, args['d_a'])
        self.linear_second_seq = torch.nn.Linear(args['d_a'], self.r)

        # CNN components
        self.conv = conv3x3(1, 8)  # Initial conv layer with 8 channels
        self.bn = nn.BatchNorm2d(8)
        self.elu = nn.ELU(inplace=False)
        
        # Layer 1: 8 -> 16 channels
        self.layer1 = self._make_layer(block, 16, args['cnn_layers'], 8)
        
        # Layer 2: 16 -> 32 channels
        self.layer2 = self._make_layer(block, 32, args['cnn_layers'], 16)
        
        # Layer 3: 32 -> 32 channels
        self.layer3 = self._make_layer(block, 32, args['cnn_layers'], 32)
        
        # Final layers
        self.linear_final_step = torch.nn.Linear(self.lstm_hid_dim*2+args['d_a'], args['dense_hid'])
        self.linear_final = torch.nn.Linear(args['dense_hid'], args['n_classes'])
        
        self.hidden_state = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(4,self.batch_size,self.lstm_hid_dim).to(device)),
                Variable(torch.zeros(4,self.batch_size,self.lstm_hid_dim).to(device)))

    def _make_layer(self, block, out_channels, blocks, in_channels):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels),
                nn.BatchNorm2d(out_channels))
        
        layers = []
        layers.append(block(in_channels, out_channels, 1, downsample))
        
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
            
        return nn.Sequential(*layers)

    # Alias for backward compatibility
    make_layer = _make_layer

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)

    def forward(self,x1,x2):
        # Process SMILES sequence
        smile_embed = self.embeddings(x1)
        outputs, self.hidden_state = self.lstm(smile_embed, self.hidden_state)
        sentence_att = F.tanh(self.linear_first(outputs))
        sentence_att = self.linear_second(sentence_att)
        sentence_att = self.softmax(sentence_att,1)
        sentence_att = sentence_att.transpose(1,2)
        sentence_embed = sentence_att@outputs
        avg_sentence_embed = torch.sum(sentence_embed,1)/self.r

        # Process contact map through CNN
        pic = self.conv(x2)
        pic = self.bn(pic)
        pic = self.elu(pic)
        pic = self.layer1(pic)  # 8 -> 16 channels
        pic = self.layer2(pic)  # 16 -> 32 channels
        pic = self.layer3(pic)  # 32 -> 32 channels
        pic_emb = torch.mean(pic,2)
        pic_emb = pic_emb.permute(0,2,1)
        
        # Process CNN output
        seq_att = F.tanh(self.linear_first_seq(pic_emb))
        seq_att = self.linear_second_seq(seq_att)
        seq_att = self.softmax(seq_att,1)
        seq_att = seq_att.transpose(1,2)
        seq_embed = seq_att@pic_emb
        avg_seq_embed = torch.sum(seq_embed,1)/self.r

        # Combine and process final output
        sscomplex = torch.cat([avg_sentence_embed,avg_seq_embed],dim=1)
        sscomplex = F.relu(self.linear_final_step(sscomplex))
        
        if not bool(self.type):
            output = F.sigmoid(self.linear_final(sscomplex))
            return output,seq_att
        else:
            return F.log_softmax(self.linear_final(sscomplex)),seq_att