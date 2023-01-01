import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable

from torch.autograd import Variable

from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DeepEBlock(torch.nn.Module):
    def __init__(self, input_dim,output_dim,hidden_drop,activation=torch.nn.ReLU,layers=2,identity_drop=0):
        super(DeepEBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation()
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.final_bn = torch.nn.BatchNorm1d(output_dim)
        self.identity_bn = torch.nn.BatchNorm1d(output_dim)
        self.identity_drop = None
        if identity_drop != 0:
            self.identity_drop = torch.nn.Dropout(identity_drop)
        
        assert(layers>=2)
        
        self.reslayer = torch.nn.Sequential(
            torch.nn.Linear(input_dim,output_dim),
            torch.nn.BatchNorm1d(output_dim)
        )
        
        self.reslayer.append(self.hidden_drop)
        self.reslayer.append(activation())
        
        for i in range(layers-1):
            self.reslayer.append(torch.nn.Linear(output_dim,output_dim))            
            self.reslayer.append(torch.nn.BatchNorm1d(output_dim))
            self.reslayer.append(self.hidden_drop)
            if i != (layers-2):
                self.reslayer.append(activation())
                
        if input_dim != output_dim:
            self.dim_map = torch.nn.Linear(input_dim,output_dim)
        
            
    def forward(self,x):
        if self.input_dim != self.output_dim:
            identity = self.dim_map(x)
            identity = self.hidden_drop(identity)
            identity = self.identity_bn(identity)
        else:
            identity = x
        if (self.input_dim == self.output_dim) and self.identity_drop != 0:
            identity = self.identity_drop(identity)
        x = identity + self.reslayer(x)
        x = self.final_bn(x)
        return x
class ResNetBlock(torch.nn.Module):
    def __init__(self, input_dim,output_dim,hidden_drop,activation=torch.nn.ReLU,layers=2):
        super(ResNetBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation()
        
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.final_bn = torch.nn.BatchNorm1d(output_dim)
        self.identity_bn = torch.nn.BatchNorm1d(output_dim)
        self.hidden_drop = None
        if hidden_drop != 0:
            self.hidden_drop = torch.nn.Dropout(hidden_drop)
        
        
        assert(layers>=2)
        
        self.reslayer = torch.nn.Sequential(
            torch.nn.Linear(input_dim,output_dim),
            torch.nn.BatchNorm1d(output_dim)
        )
        
        self.reslayer.append(activation())
        
        for i in range(layers-1):
            self.reslayer.append(torch.nn.Linear(output_dim,output_dim))            
            self.reslayer.append(torch.nn.BatchNorm1d(output_dim))
            
            if i != (layers-2):
                self.reslayer.append(activation())
                
        if input_dim != output_dim:
            self.dim_map = torch.nn.Linear(input_dim,output_dim)
        
            
    def forward(self,x):
        if self.input_dim != self.output_dim:
            identity = self.dim_map(x)
        else:
            identity = x
        x = identity + self.reslayer(x)
        x = self.final_bn(x)
        
        if self.hidden_drop is not None:
            x = self.hidden_drop(x)
        x = self.activation(x)
        
        return x
    
    

class DeepE(torch.nn.Module):
    def __init__(self, num_emb,embedding_dim=300,hidden_drop=0.4,num_source_layers=5,num_target_layers=1,input_drop = 0.4,inner_layers=3,target_drop=0,identity_drop=0):
        super(DeepE, self).__init__()
        self.emb = torch.nn.Embedding(num_emb, embedding_dim)
        self.num_source_layers = num_source_layers
        self.num_target_layers = num_target_layers
        self.input_drop = torch.nn.Dropout(input_drop)
        self.input_bn = torch.nn.BatchNorm1d(2*embedding_dim)
        self.target_bn = torch.nn.BatchNorm1d(embedding_dim)
        self.loss = torch.nn.CrossEntropyLoss()
        self.source_layers = torch.nn.Sequential()
        self.target_layers = torch.nn.Sequential()
        for i in range(num_source_layers):
            if i ==0:
                input_emb = embedding_dim*2
            else:
                input_emb = embedding_dim
            self.source_layers.append(DeepEBlock(input_emb,embedding_dim,hidden_drop,torch.nn.ReLU,layers=2,identity_drop=identity_drop))
        
        for i in range(num_target_layers):
            self.target_layers.append(ResNetBlock(embedding_dim,embedding_dim,target_drop,torch.nn.ReLU,layers=inner_layers))
        
        self.register_parameter('b', Parameter(torch.zeros(num_emb)))
        
        
    def to_var(self, x, use_gpu=True):
        if use_gpu:
            return Variable(torch.from_numpy(x).long().cuda())
        
    def init(self):
        xavier_normal_(self.emb.weight.data)
        self.emb.weight.data = self.emb.weight.data

    
    def forward(self, e1, rel):
        e1 = self.to_var(e1)
        rel = self.to_var(rel)
        
        e1_embedded= self.emb(e1)
        rel_embedded = self.emb(rel)
        
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], -1)

        x = self.input_bn(stacked_inputs)
        x = self.input_drop(x)
        
        x = self.source_layers(x)
        
        weight = self.emb.weight
        
        
        weight = self.target_bn(weight)
        weight = self.target_layers(weight)
        #weight = self.input_drop(weight)
        weight = weight.transpose(1,0)
        x = torch.mm(x, weight)
        x += self.b.expand_as(x)
        pred = x
        
        return pred