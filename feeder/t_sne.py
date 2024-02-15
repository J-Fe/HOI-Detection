import torch
import numpy as np
import joblib
from PIL import Image
import gtransforms as gtransforms
from explaned_explanation import chatgpt_sample_expanded_explanation, chatgpt_affordence_list

'''
####################################################
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
human_list = [i for i in sample_expanded_explanation.values()] ###### Wikepieda 바꿔야댐
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


from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

import seaborn as sns
from matplotlib import pyplot as plt
'''
data = load_digits()

n_compnents =2

model = TSNE(n_components=n_compnents)

X_embedded = model.fit_transform(data.data)


print(X_embedded[0])

palette = sns.color_palette("bright", 10)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=data.target, legend='full', palette=palette)
plt.show()
'''
digits = load_digits()
X = digits.data
y = digits.target

    # t-SNE(64 to 2)
tsne = TSNE(n_components=2, random_state=1)
X_tsne = tsne.fit_transform(X)

    # t-SNE Visualization
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.show()