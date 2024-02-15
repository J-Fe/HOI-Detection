import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import numpy as np
import argparse
import time
from tqdm import tqdm
import sklearn.metrics
import random
random.seed(0)
from models.STIGPN import VisualModelV
from models.STIGPN import SemanticModelV
import pickle
#torch.backends.cudnn.benchmark = True
from feeder.dataset import Dataset
def run_model(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if args.model == 'VisualModelV':
        model = VisualModelV(args)
    elif args.model == 'SemanticModelV':
        model = SemanticModelV(args)
    # calculate the amount of all the learned parameters
    parameter_num = 0
    for param in model.parameters():
        parameter_num += param.numel()
    print(f'The parameters number of the model is {parameter_num / 1e6} million')

    model.float().cuda()
    learning_rate = args.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load pre_process data from files
    if args.task == 'Detection':
        train_dataset = Dataset(args,is_val=False,isAnticipation=False)
        val_dataset = Dataset(args,is_val=True,isAnticipation=False)
    else:
        train_dataset = Dataset(args,is_val=False,isAnticipation=True)
        val_dataset = Dataset(args,is_val=True,isAnticipation=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    print('set up dataloader successfully')
    
    for epoch in range(args.start_epoch, args.epoch):
        
        train(epoch,args,model,train_dataloader,criterion,optimizer)
        
        '''
        if (epoch+1) % args.eval_every == 0:
            eval(epoch,args, model, val_dataloader,criterion)
        if (epoch+1)%args.step_size == 0:
            if learning_rate > 1e-7:
                learning_rate *= args.weight_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                print('decrease lr to', learning_rate)
        '''
        

def train(epoch,args,model, train_dataloader,criterion,optimizer):
    start_time = time.time()
    model.train()
    model.zero_grad()
    total_loss, total_human_loss, total_obj_loss = 0.0, 0.0, 0.0
    H_preds, H_gts, O_preds, O_gts = [], [], [], []
    
    
    deep_features = []
    actual = []
    
    for num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label in tqdm(train_dataloader):
        
        
        
        '''
        ###################################
        # TSNE 만들기 위한 Feature 정제 작엄 
    
         #<concat - spatial and appearance>
        embedding_feature_dim = 256
        res_feat_dim = 2048
        preprocess_dim = 1024
        coord_to_feature = nn.Sequential(
            nn.Linear(4, embedding_feature_dim//2, bias=False),
            nn.BatchNorm1d(embedding_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_feature_dim//2, embedding_feature_dim, bias=False),
            nn.BatchNorm1d(embedding_feature_dim),
            nn.ReLU()
        )
        appearence_preprocess = nn.Linear(res_feat_dim, preprocess_dim)
        appearence_feats = appearence_preprocess(appearance_feats.reshape(1*6*10,res_feat_dim))#(180,1024)
        box_input = box_tensors.transpose(2, 1).contiguous()
        box_input = box_input.view(1*6*10, 4)
        #print(box_input.size()) #(180,4)
        spatial_feats = coord_to_feature(box_input)
        appearence_spatial_feats = torch.cat([spatial_feats, appearence_feats], dim=1) #(60,1280)
        appearence_spatial_feats = appearence_spatial_feats.reshape(6,10,1280)
        appearence_spatial_feats = torch.sum(appearence_spatial_feats[0], dim = 0)
        appearence_spatial_feats = appearence_spatial_feats.unsqueeze(0)
        

        
        ##여기만 열면됨
        features = torch.sum(appearance_feats[0][0], dim = 0)
        features = features.unsqueeze(0) # (1,2048)
        
        #features = appearance_feats.reshape(1,-1)
        #features = appearance_feats[0][0]
        
        deep_features += features.cpu().detach().numpy().tolist()
        actual += sub_activity_label.cpu().detach().numpy().tolist()
        
        '''
        
        ####################################
        
        
        
        
        
        loss,human_loss,object_loss,h_preds,h_gts,o_preds,o_gts = \
            forward_step(args, model, criterion,num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label,True,optimizer=optimizer)

        total_loss += loss.item()
        total_human_loss += human_loss.item()
        total_obj_loss += object_loss.item() 

        H_preds += h_preds
        O_preds += o_preds
        H_gts += h_gts
        O_gts += o_gts

    total_loss = total_loss / len(train_dataloader)
    total_human_loss = total_human_loss / len(train_dataloader)
    total_obj_loss = total_obj_loss / len(train_dataloader)

    H_gts = list(map(int, H_gts)) 
    O_gts = list(map(int, O_gts)) 

    human_accuracy = 0.0
    for i in range(len(H_gts)):
        if H_gts[i] == H_preds[i]:
            human_accuracy += 1.0
    human_accuracy = 100.0*human_accuracy/len(H_gts)

    object_accuracy = 0.0
    for i in range(len(O_gts)):
        if O_gts[i] == O_preds[i]:
            object_accuracy += 1.0
    object_accuracy = 100.0*object_accuracy/len(O_gts)
    
    end_time = time.time()
    print('Epoch:%02d, loss: %.6f, human_loss: %.6f, object_loss: %.6f, human_acc: %.4f, object_acc: %.4f, time: %.3f s/iter' %
          (epoch, total_loss, total_human_loss, total_obj_loss, human_accuracy, object_accuracy, (end_time-start_time)))
max_scores = 0,0
min_loss = 10

'''
    with open('origin_apperance_10_2048','wb') as f:
        pickle.dump(deep_features,f)
        
    with open('origin_list_10_2048','wb') as f:
        pickle.dump(actual, f)

'''



def eval(epoch,args, model, val_dataloader,criterion):
    start_time = time.time()
    model.eval()
    total_loss, total_human_loss, total_obj_loss = 0.0, 0.0, 0.0
    H_preds, H_gts, O_preds, O_gts = [], [], [], []
    for num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label in tqdm(val_dataloader):
        

        loss,human_loss,object_loss,h_preds,h_gts,o_preds,o_gts = \
            forward_step(args, model, criterion,num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label,False,optimizer=None)
        
        total_loss += loss.item()
        total_human_loss += human_loss.item()
        total_obj_loss += object_loss.item() 

        H_preds += h_preds
        O_preds += o_preds
        H_gts += h_gts
        O_gts += o_gts
    
    total_loss = total_loss / len(val_dataloader)
    total_human_loss = total_human_loss / len(val_dataloader)
    total_obj_loss = total_obj_loss / len(val_dataloader)

    H_gts = list(map(int, H_gts)) 
    O_gts = list(map(int, O_gts))
    subact_f1_score = sklearn.metrics.f1_score( H_gts, H_preds, labels=range(10), average='macro')*100
    afford_f1_score = sklearn.metrics.f1_score( O_gts, O_preds, labels=range(12), average='macro')*100
    
    end_time = time.time()
    print('Test   ' + \
        ', loss: %.6f'% total_loss + \
        ', human_loss: %.6f'% total_human_loss + \
        ', obj_loss: %.6f'% total_obj_loss + \
        ', subact_fmacro: %.5f'%(subact_f1_score) + \
        ', afford_fmacro: %.5f'%(afford_f1_score))
    global max_scores
    if max_scores[0]+max_scores[1] < round(subact_f1_score,2) + round(afford_f1_score,2):
        max_scores = round(subact_f1_score,2),round(afford_f1_score,2)
        if args.task == 'Detection':
            torch.save(model.state_dict(),'checkpoints/0816'+ 'wiki-MHlayer1' +'mh1layer_Semantic.pkl') #########################args.model
        else:
            torch.save(model.state_dict(),'checkpoints/0816/'+'wiki-MHlayer1'+'mh1layer_Semantic.pkl') #########################args.model + args.task
    print('TOP:',max_scores)
    
    #위에 세이브 수정해서 훈련시켜야댐

def forward_step(args, model, criterion,num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label,isTrain,optimizer):
    batchSize = len(num_objs)
    appearance_feats = appearance_feats.cuda()
    box_tensors = box_tensors.cuda()
    box_categories = box_categories.cuda()
    valid_labels = []
    for b in range(batchSize):
        for n in range(0, num_objs[b]):
            valid_labels.append(affordence_label[b][n])
    affordence_label = torch.Tensor(valid_labels)
    sub_activity_label,affordence_label = sub_activity_label.cuda(),affordence_label.cuda()

    if isTrain:
        subact_cls_scores, afford_cls_scores = model(num_objs,appearance_feats,box_tensors,box_categories)
    else:
        with torch.no_grad():
            subact_cls_scores, afford_cls_scores = model(num_objs,appearance_feats,box_tensors,box_categories)
    
    
    human_loss = criterion(subact_cls_scores, sub_activity_label.long())
    object_loss = criterion(afford_cls_scores, affordence_label.long())
    loss = human_loss + args.obj_scal*object_loss

    if isTrain:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    subact_cls_scores = subact_cls_scores.cpu().detach().numpy()
    afford_cls_scores = afford_cls_scores.cpu().detach().numpy()

    h_preds = []
    h_gts = []
    for b in range(batchSize):
        H_pred = np.argmax(subact_cls_scores[b])
        h_preds.append(H_pred)
        h_gts.append(sub_activity_label.cpu().numpy()[b])

    o_preds = []
    o_gts = []
    for b in range(affordence_label.shape[0]):
        O_pred = np.argmax(afford_cls_scores[b])
        o_preds.append(O_pred)
        o_gts.append(affordence_label.cpu().numpy()[b].item())

    return loss,human_loss,object_loss,h_preds,h_gts,o_preds,o_gts

parser = argparse.ArgumentParser(description="You Can Do It!")
parser.add_argument('--model', default='VisualModelV',help='VisualModelV,SemanticModelV') 
parser.add_argument('--task', default='Detection')
parser.add_argument('--batch_size', '--b_s', type=int, default=1, help='batch size: 1') ############################ 원래배치 사이즈 3이었음
parser.add_argument('--start_epoch', type=int, default=0,help='number of beginning epochs : 0')
parser.add_argument('--epoch', type=int, default=1,help='number of epochs to train: 300')############################## 원래 epoch 300이었음 
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate: 0.0001')#2e-5
parser.add_argument('--weight_decay', type=float, default=0.8, help='learning rate: 0.0001')
parser.add_argument('--nr_boxes', type=int, default=6,help='number of bbox : 6')
parser.add_argument('--nr_frames', type=int, default=10,help='number of frames : 10')
parser.add_argument('--subact_classes', type=int, default=10,help='number of subact_classes : 10')
parser.add_argument('--afford_classes', type=int, default=12,help='number of afford_classes : 12')
parser.add_argument('--feat_drop', type=float, default=0,help='dropout parameter: 0')
parser.add_argument('--attn_drop', type=float, default=0,help='dropout parameter: 0')
parser.add_argument('--cls_dropout', type=float, default=0,help='dropout parameter: 0')
parser.add_argument('--step_size', type=int, default=50,help='number of steps for validation loss: 10') 
parser.add_argument('--eval_every', type=int, default=1,help='number of steps for validation loss: 10') 
parser.add_argument('--obj_scal', type=int, default=1,help='number of steps for validation loss: 10') 
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    run_model(args)
