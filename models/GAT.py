import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from dgl.nn import GATConv, AvgPooling, MaxPooling, GATv2Conv

class unit_GAT(nn.Module):
    def __init__(self, in_channels, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(unit_GAT, self).__init__()
        self.gat = GATConv(in_feats=in_channels, out_feats=out_channels, num_heads=1,feat_drop=feat_drop,attn_drop=attn_drop,residual=True,activation=activation)##########original num_heads = 1
        
        
        
        #num_heads 를 왜 1밖에 안하는거지. 이거 쫌 이해안되네 이친구
        ##################
        #self.gat2 = GATConv(in_feats=out_channels*4, out_feats=out_channels, num_heads=1,feat_drop=feat_drop,attn_drop=attn_drop,residual=True,activation=activation)
        #self.avgpooling = AvgPooling()
        #################
        
        
        
    def forward(self, graph, x):
        #batch_size,nodes,frame,feature_size = x.shape
        #y = x.reshape(batch_size*nodes*frame,feature_size)
        
        #bh = x.shape[0] ##########
  
        
        y = self.gat(graph,x) # foward에 얘만 있었음 원래 
       
        
        
        #y = F.relu(y) ########### relu넣어야하나. 
        #y = y.reshape(bh, -1)############# (180, 1024) 
        
        #y = self.gat2(graph,y)###############
        
        #y = y.reshape(batch_size,nodes,frame,-1)
        
        return y

class GAT(nn.Module):
    def __init__(self, in_channels, nhidden, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(GAT, self).__init__()

        #self.l1 = unit_GAT(in_channels, in_channels,feat_drop,attn_drop,activation)
        self.l2 = unit_GAT(in_channels, out_channels,feat_drop,attn_drop,activation)

    def forward(self, graph, x):
        #x = self.l1(graph, x)
        x = self.l2(graph, x)
        return x


class unit_GAT2(nn.Module):
    def __init__(self, in_channels, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(unit_GAT2, self).__init__()
        self.gat = GATConv(in_feats=in_channels, out_feats=out_channels, num_heads=8,feat_drop=feat_drop,attn_drop=attn_drop,residual=True,activation=activation)##########original num_heads = 1
        
        
        
        #num_heads 를 왜 1밖에 안하는거지. 이거 쫌 이해안되네 이친구
        ##################
        self.gat2 = GATConv(in_feats=out_channels*8, out_feats=out_channels, num_heads=1,feat_drop=feat_drop,attn_drop=attn_drop,residual=True,activation=activation)
        self.avgpooling = AvgPooling()
        #################
        
        
        
    def forward(self, graph, x):
        #batch_size,nodes,frame,feature_size = x.shape
        #y = x.reshape(batch_size*nodes*frame,feature_size)
        
        bh = x.shape[0] ##########
  
        
        y = self.gat(graph,x) # foward에 얘만 있었음 원래 
       
        
        
        y = F.relu(y) ########### relu넣어야하나. 
        y = y.reshape(bh, -1)############# (180, 1024) 
        
        y = self.gat2(graph,y)###############
        
        #y = y.reshape(batch_size,nodes,frame,-1)
        
        return y

class GAT2(nn.Module):
    def __init__(self, in_channels, nhidden, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(GAT2, self).__init__()

        #self.l1 = unit_GAT(in_channels, in_channels,feat_drop,attn_drop,activation)
        self.l2 = unit_GAT2(in_channels, out_channels,feat_drop,attn_drop,activation)

    def forward(self, graph, x):
        #x = self.l1(graph, x)
        x = self.l2(graph, x)
        return x
    
    
    
class unit_GAT3(nn.Module):
    def __init__(self, in_channels, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(unit_GAT3, self).__init__()
        self.gat = GATv2Conv(in_feats=in_channels, out_feats=out_channels, num_heads=8,feat_drop=feat_drop,attn_drop=attn_drop,residual=True,activation=activation)##########original num_heads = 1
        
        
        
        #num_heads 를 왜 1밖에 안하는거지. 이거 쫌 이해안되네 이친구
        ##################
        self.gat2 = GATv2Conv(in_feats=out_channels*8, out_feats=out_channels, num_heads=1,feat_drop=feat_drop,attn_drop=attn_drop,residual=True,activation=activation)
        self.avgpooling = AvgPooling()
        #################
        
        
        
    def forward(self, graph, x):
        #batch_size,nodes,frame,feature_size = x.shape
        #y = x.reshape(batch_size*nodes*frame,feature_size)
        
        bh = x.shape[0] ##########
  
        
        y = self.gat(graph,x) # foward에 얘만 있었음 원래 
       
        
        
        y = F.relu(y) ########### relu넣어야하나. 
        y = y.reshape(bh, -1)############# (180, 1024) 
        
        y = self.gat2(graph,y)###############
        
        #y = y.reshape(batch_size,nodes,frame,-1)
        
        return y

class GATV2(nn.Module):
    def __init__(self, in_channels, nhidden, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(GATV2, self).__init__()

        #self.l1 = unit_GAT(in_channels, in_channels,feat_drop,attn_drop,activation)
        self.l2 = unit_GAT3(in_channels, out_channels,feat_drop,attn_drop,activation)

    def forward(self, graph, x):
        #x = self.l1(graph, x)
        x = self.l2(graph, x)
        return x
    
    
    
class unit_GAT4(nn.Module):
    def __init__(self, in_channels, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(unit_GAT4, self).__init__()
        self.gat = GATv2Conv(in_feats=in_channels, out_feats=out_channels, num_heads=8,feat_drop=feat_drop,attn_drop=attn_drop,residual=True,activation=activation)##########original num_heads = 1
        
        
        
        #num_heads 를 왜 1밖에 안하는거지. 이거 쫌 이해안되네 이친구
        ##################
        self.gat2 = GATv2Conv(in_feats=out_channels*8, out_feats=out_channels, num_heads=1,feat_drop=feat_drop,attn_drop=attn_drop,residual=True,activation=activation)
        self.avgpooling = AvgPooling()
        #################
        
        
        
    def forward(self, graph, x):
        #batch_size,nodes,frame,feature_size = x.shape
        #y = x.reshape(batch_size*nodes*frame,feature_size)
        
        bh = x.shape[0] ##########
  
        
        y = self.gat(graph,x) # foward에 얘만 있었음 원래 
       
        
        
        y = F.relu(y) ########### relu넣어야하나. 
        y = y.reshape(bh, -1)############# (180, 1024) 
        
        y = self.gat2(graph,y)###############
        
        #y = y.reshape(batch_size,nodes,frame,-1)
        
        return y

class GATV3(nn.Module):
    def __init__(self, in_channels, nhidden, out_channels,feat_drop=0,attn_drop=0,activation=None):
        super(GATV3, self).__init__()

        #self.l1 = unit_GAT(in_channels, in_channels,feat_drop,attn_drop,activation)
        self.l2 = unit_GAT4(in_channels, out_channels,feat_drop,attn_drop,activation)

    def forward(self, graph, x):
        #x = self.l1(graph, x)
        x = self.l2(graph, x)
        return x