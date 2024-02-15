import torch
import numpy as np
import joblib
from PIL import Image
import feeder.gtransforms as gtransforms
from feeder.explaned_explanation import chatgpt_sample_expanded_explanation, chatgpt_affordence_list

###################################################
# 1. ROI crop은 관심영역을 짤랐으면 해상도를 224x224x3으로 고정시킨것
# 2. 카테고리는 one-hot encoding으로 각 객체별 수행해서 텐서로 저장
# 3. MOTraking 사용해서 박스 위치 및 센터 포인트 추출 
# 4. Faster-RCNN은 미리 category별로 학습 시킴. 
###################################################

'''
###################################################
from transformers import AutoTokenizer
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
tokenizer =  AutoTokenizer.from_pretrained('bert-base-uncased')

bertmodel = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=True, output_hidden_states=True, output_attentions=True)

pooling_method = nn.Sequential(
    nn.Linear(768,768*2),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(768*2, 2048)
)

#chatpbot list version
chatgpt_sample_expanded_explanation = chatgpt_sample_expanded_explanation
chatgpt_affordence_list = chatgpt_affordence_list

sample_expanded_explanation = {'reaching' : 'Action. When somebody is taking something too far', 'eating':'to take food into your mouth and swallow it',
                                'cleaning':'to make (something) clean; to remove dirt, marks, etc., from (something)','closing':' to move (a door, window, etc.) so that things cannot pass through an opening',
                                'drinking':'to take a liquid into your mouth and swallow it','moving':'to go from one place or position to another','null':'to make (something) legally null',
                                'opening':' to move (a door, window, etc.) so that an opening is no longer covered','placing':'to put (something or someone) in a particular place or position',
                                'pouring':'to cause (something) to flow in a steady stream from or into a container or place'}

sub_activity_list =  ['reaching', 'eating', 'cleaning', 'closing', 'drinking', 'moving','null', 'opening', 'placing', 'pouring']


affordence_list = {'stationary':'not moving, staying in one place or position', 'cleanable':'That can be cleaned', 'cleaner':'a substance used for cleaning things', 
                    'closeable':'That can be closed.', 'containable':'Able to be contained, especially applied to viruses and similar diseases', 'drinkable':'able to be drunk', 'movable':'able to be moved opposite', 
                    'openable':'Capable of being opened.', 'placeable':'Something which is conventionally associated with a specific place; for example, blinds go on windows, carpets go on floors.', 
                    'pourable':'Liquid, with viscosity allowing to be poured, or solid with particles not strongly adhering to each other or the container. ', 'pourto':' to cause (something) to flow in a steady stream from or into a container or place', 
                    'reachable':'(countable, mathematics) The extent to which a node in a graph is reachable from others '}

obj_affordence_list = ['stationary', 'cleanable', 'cleaner', 'closeable', 'containable', 'drinkable', 'movable', 'openable', 'placeable', 'pourable', 'pourto', 'reachable']

#인간과 객체의 행동 클래스에 대한 expandeed explanation
#human_list = [i for i in sample_expanded_explanation.values()]
#affordence_list = [i for i in affordence_list.values()]

#chatbot list
human_list = [i for i in chatgpt_sample_expanded_explanation.values()] ###### Wikepieda 바꿔야댐
affordence_list = [i for i in affordence_list.values()]

#인간과 객체 피쳐를 추출해놓은 텐서
human_tensor_list_features = torch.zeros(len(human_list),1,10,2048) # 10은 프레임 리스트
object_affordance_list_features = torch.zeros(len(affordence_list),1,10,2048)


tokens = len(tokenizer.tokenize(human_list[0]))
tokens2 = len(tokenizer.tokenize(human_list[1]))

tokens_human_list = []
tokens_object_list = []

for x in range(len(human_list)):
    i = len(tokenizer.tokenize(human_list[x]))
    tokens_human_list.append(int(i))    

for x in range(len(affordence_list)):
    i = len(tokenizer.tokenize(affordence_list[x]))
    tokens_object_list.append(int(i))


#인간 행동 관련된 피쳐를 저장
for x in range(len(human_list)):
    tokens = tokenizer(
        human_list[x],
        max_length= tokens_human_list[x],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    

    text_embedding = bertmodel(**tokens)
    text_embedding = text_embedding.last_hidden_state
    text_embedding = pooling_method(text_embedding)
    text_embedding_sum  = torch.sum(text_embedding, dim = 1).unsqueeze(1)
    text_embedding_sum_10 = text_embedding_sum
    #text_embedding_sum = F.normalize(text_embedding_sum, dim = 1)
    for i in range(9):
        text_embedding_sum_10 = torch.concat([text_embedding_sum_10,text_embedding_sum],dim = 1)
    
    human_tensor_list_features[x,:]= text_embedding_sum_10

#객체 행동 관련된 피쳐를 저장
for x in range(len(affordence_list)):
    tokens = tokenizer(
        affordence_list[x],
        max_length = tokens_object_list[x],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    text_embedding = bertmodel(**tokens)
    text_embedding = text_embedding.last_hidden_state
    text_embedding = pooling_method(text_embedding)
    text_embedding_sum  = torch.sum(text_embedding, dim = 1).unsqueeze(1)
    text_embedding_sum_10 = text_embedding_sum
    #text_embedding_sum = F.normalize(text_embedding_sum, dim = 1)
    for i in range(9):
        text_embedding_sum_10 = torch.concat([text_embedding_sum_10,text_embedding_sum],dim = 1)
    object_affordance_list_features[x,:]= text_embedding_sum_10


########################################################################
'''


class Dataset(torch.utils.data.Dataset):
    def __init__(self,args,is_val=False,isSegment=False): ####### Eval 할때는 Segment Ture 로 바꿔야함
        self.is_val = is_val
        self.num_boxes = args.nr_boxes
        self.coord_nr_frames = args.nr_frames
        self.pre_resize_shape = (640, 480)
        self.if_augment = True
        self.segment = isSegment

        if self.if_augment:
            if not self.is_val:  # train, multi scale cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=(640,480),
                                                                   scales=[1, .875, .75])
            else:  # val, only center cropping
                self.random_crop = gtransforms.GroupMultiScaleCrop(output_size=(640,480),
                                                                   scales=[1],
                                                                   max_distort=0,
                                                                   center_crop_only=True)
        else:
            self.random_crop = None

        if not is_val:
            load_dir = 'datasets/cad_train_data_with_appearence_features.p'
        else:
            load_dir = 'datasets/cad_test_data_with_appearence_features.p'
        with open(load_dir,'rb') as f:
            data = joblib.load(f)
        self.sub_activity_list = data['sub_activity_list']
        self.affordence_list = data['affordence_list']
        self.classes_list = data['classes_list']
        self.classes_list.insert(0,'person')
        self.classes_list.insert(0,'_background_')
        if not is_val:
            self.load_data = data['train_data']
        else:
            self.load_data = data['test_data']


    def __len__(self):
        return len(self.load_data)

    def _sample_indices(self, nr_video_frames): #37
        average_duration = nr_video_frames * 1.0 / self.coord_nr_frames #37/10
        if average_duration > 0:
            offsets = np.multiply(list(range(self.coord_nr_frames)), average_duration) \
                      + np.random.uniform(0, average_duration, size=self.coord_nr_frames)
            offsets = np.floor(offsets)
        elif nr_video_frames > self.coord_nr_frames:
            offsets = np.sort(np.random.randint(nr_video_frames, size=self.coord_nr_frames))
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets

    def _get_val_indices(self, nr_video_frames):
        if nr_video_frames > self.coord_nr_frames:
            tick = nr_video_frames * 1.0 / self.coord_nr_frames
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.coord_nr_frames)])
        else:
            offsets = np.zeros((self.coord_nr_frames,))
        offsets = list(map(int, list(offsets)))
        return offsets
    
    def _cn_dataset(self):
        pre_video_id = -1
        num_objs_list,appearance_feats_list,box_tensors_list,box_categories_list,sub_activity_label_list,affordence_label_list = \
            [],[],[],[],[],[]
        for i in range(0,self.__len__()):
            video_id,seg_frames,num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label = \
                self.__getitem__(i)
            if pre_video_id == -1:
                pre_video_id = video_id
            if pre_video_id != video_id or i == self.__len__() - 1:
                num_objs_ = torch.tensor(num_objs_list)
                appearance_feats_ = torch.cat(appearance_feats_list,dim=0)
                box_tensors_ = torch.cat(box_tensors_list,dim=0)
                box_categories_ = torch.cat(box_categories_list,dim=0)
                sub_activity_label_ = torch.cat(sub_activity_label_list,dim=0)
                affordence_label_ = torch.cat(affordence_label_list,dim=0)
                yield num_objs_,appearance_feats_,box_tensors_,box_categories_,sub_activity_label_,affordence_label_
                pre_video_id = video_id
                num_objs_list,appearance_feats_list,box_tensors_list,box_categories_list,sub_activity_label_list,affordence_label_list = \
                    [],[],[],[],[],[]
            num_objs_list.append(num_objs)
            appearance_feats_list.append(appearance_feats.unsqueeze(0))
            box_tensors_list.append(box_tensors.unsqueeze(0))
            box_categories_list.append(box_categories.unsqueeze(0))
            sub_activity_label_list.append(sub_activity_label.unsqueeze(0))
            affordence_label_list.append(affordence_label.unsqueeze(0))
           

    
    def __getitem__(self, i):
        sub_activity_feature = self.load_data[i]
        video_id = sub_activity_feature['video_id'] #string number : 0510173506
        seg_frames = sub_activity_feature['seg_frames'] #tuple ex) (309,334)
        label = sub_activity_feature['label'] #dict ex) {'person': 'eating', 'cup_1': 'stationary'}
        bboxes = sub_activity_feature['bboxes'] # dict 객체가 4개까지도 가능하네
        apperence_features = sub_activity_feature['apperence_features'] # numpy (6,84,2048) 피쳐 사이즈. 84부분은 각 비디오영상의 프레임 
        # 그니까 6개의 

        '''
        #############################################################################################################
        #각 객체들의 레이블 밸류를 리스트화
        object_label = [x for x in label.values()]

        #총 레이블 샘플 리스트
        sample_index_list = []
        sample_index_list.append(sub_activity_list.index(object_label[0]))
        for x in range(1,len(object_label)):
            sample_index_list.append(obj_affordence_list.index(object_label[x]))

        #인간과 object 인덱스 리스트
        object_index_list = []
        for x in range(1,len(object_label)):
            object_index_list.append(obj_affordence_list.index(object_label[x]))

        #매 데이터셋의 객체 텍스트 레이블을 위한 텐서 
        Object_text_tensor = torch.zeros(len(sample_index_list),10,2048)

        #인간 피쳐 먼저 텐서에 저장
        Object_text_tensor[0:1,:] = human_tensor_list_features[sample_index_list[0] , :]


        #객체 피쳐 텐서에 저장
        for index,number in enumerate(object_index_list):
            Object_text_tensor[index:index + 1,:] = object_affordance_list_features[number, :]
        ################################################################################################################
        '''
        
        object_set = [x[::-1] for x in label.keys()] #아 ㅋㅋ 글자 뒤집어서그냥 소팅할려고.. ㅇㅋㅇㅋ 1번객체와 2번객체를 구분하기 위해서.    
        object_set = sorted(object_set)
        object_set = [x[::-1] for x in object_set]
        object_set.remove('person')
        
        sub_activity_label = torch.tensor(self.sub_activity_list.index(label['person'])).float()
        affordence_label = torch.tensor([self.affordence_list.index(label[x]) for x in object_set]).float()
        affordence_label = torch.cat([affordence_label,torch.zeros((5-affordence_label.shape[0])).float()],dim=0)
        
        object_set.insert(0,'person')
        n_frame = len(bboxes)
  


        if not self.is_val:  # train
            coord_frame_list = self._sample_indices(n_frame) #37
        else:  # val
            coord_frame_list = self._get_val_indices(n_frame)
        
        ####################################
        
       


        frames = []
        frames.append(Image.new('RGB',(640,480)))
        height, width = frames[0].height, frames[0].width

        if self.random_crop is not None:
            frames, (offset_h, offset_w, crop_h, crop_w) = self.random_crop(frames)
        else:
            offset_h, offset_w, (crop_h, crop_w) = 0, 0, self.pre_resize_shape
        #print(crop_h, crop_w)
        scale_resize_w, scale_resize_h = self.pre_resize_shape[1] / float(width), self.pre_resize_shape[0] / float(height)
        scale_crop_w, scale_crop_h = 640 / float(crop_w), 480 / float(crop_h)
        
        
        
        box_categories = torch.zeros((self.num_boxes)) #6
        box_tensors = torch.zeros((self.coord_nr_frames, self.num_boxes, 4), dtype=torch.float32) #(10, 6, 4)

        appearance_feats = torch.zeros([self.num_boxes, self.coord_nr_frames, 2048])*1.0 #(6, 10, 2048)
        
        
        
        for frame_index, frame_id in enumerate(coord_frame_list): # sample index frame id ex) [2,4,9,...39]
            
            
            try:
                frame_bboxes_data = bboxes[frame_id]
                appearance_feats[:,frame_index,:] = torch.tensor(apperence_features[:,frame_id,:]).float() 
            except:
                frame_bboxes_data = {}
            for box_data_key in frame_bboxes_data.keys():
                
                global_box_id = object_set.index(box_data_key)
                try:
                    x0, y0, x1, y1 = frame_bboxes_data[box_data_key]
                except:
                    x0, y0, x1, y1 = 0,0,0,0
                
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)  #샘플 아이디 박스 위치 정보 가지고옴
              
                # scaling due to initial resize
                x0, x1 = x0 * scale_resize_w, x1 * scale_resize_w
                y0, y1 = y0 * scale_resize_h, y1 * scale_resize_h

                # shift
                x0, x1 = x0 - offset_w, x1 - offset_w
                y0, y1 = y0 - offset_h, y1 - offset_h

                x0, x1 = np.clip([x0, x1], a_min=0, a_max=crop_w-1)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=crop_h-1)

                # scaling due to crop
                x0, x1 = x0 * scale_crop_w, x1 * scale_crop_w
                y0, y1 = y0 * scale_crop_h, y1 * scale_crop_h

                # precaution
                x0, x1 = np.clip([x0, x1], a_min=0, a_max=640)
                y0, y1 = np.clip([y0, y1], a_min=0, a_max=480)

                # (cx, cy, w, h)
                gt_box = np.array([(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0], dtype=np.float32)

                # normalize gt_box into [0, 1]
                gt_box[::2] = gt_box[::2]/640.0
                gt_box[1::2] = gt_box[1::2]/480.0

                # load box into tensor
                try:
                    box_tensors[frame_index, global_box_id] = torch.tensor(gt_box).float()
                except:
                    pass
                
        
        
        for idx,o in enumerate(object_set):
            box_categories[idx] = self.classes_list.index(o.split('_')[0])
        box_categories = torch.cat([box_categories.unsqueeze(0) for x in range(self.coord_nr_frames)],dim=0)


        num_objs = torch.tensor(len(object_set)-1)
        

        
        #print(video_id)
        #print(seg_frames)
        #print('##################num_objs')
        #print(num_objs)
        #print(sub_activity_label)
        #print(appearance_feats.shape)#(6, 10, 2048)
        #print(box_tensors.shape)#torch.Size([10, 6, 4])
        #print(box_categories.shape)#torch.Size([10, 6])
        

        
        #그니까 가볍게 생각하면은 appresnfe_features 가 6, 10 2048이고 단어가

        #appearance_feats[0:len(sample_index_list),:] * Object_text_tensor ############################################

      
        if self.segment:
            return video_id,seg_frames,num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label,label####label은 그냥 test를 위해서 뭔지 확인하려고
        return num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label 
#num_objs,appearance_feats,box_tensors,box_categories,sub_activity_label,affordence_label
if __name__ == "__main__":
    a = Dataset(is_val=False)
    for i in range(a.__len__()):
        a.__getitem__(i)
