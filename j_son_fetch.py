


import json
# 여기는 bounding box 전부 합치는거
'''
with open('/Data1/home/hshong/workspace/STIGPN/S-Else/dataset/Sth-zipfiles/bounding_box_smthsmth_part1.json','rb') as f:
    data_jsons1 = json.load(f)

with open('/Data1/home/hshong/workspace/STIGPN/S-Else/dataset/Sth-zipfiles/bounding_box_smthsmth_part2.json','rb') as f:
    data_jsons2 = json.load(f)

with open('/Data1/home/hshong/workspace/STIGPN/S-Else/dataset/Sth-zipfiles/bounding_box_smthsmth_part3.json','rb') as f:
    data_jsons3 = json.load(f)

with open('/Data1/home/hshong/workspace/STIGPN/S-Else/dataset/Sth-zipfiles/bounding_box_smthsmth_part4.json','rb') as f:
    data_jsons4 = json.load(f)


data_jsons1.update(data_jsons2)
data_jsons1.update(data_jsons3)
data_jsons1.update(data_jsons4)



with open("bounding_box_smthsmth_dict.json", 'w') as outfile:
    json.dump(data_jsons1, outfile)





with open('/Data1/home/hshong/workspace/STIGPN/S-Else/dataset/bounding_box/bouding_box_smthsmth_dict.json','rb') as f:
    data_jsons4 = json.load(f)
    

print(len(data_jsons4))
print(type(data_jsons4))



'''

'''
with open('/Data1/home/hshong/workspace/STIGPN/S-Else/dataset/annotations/compositional/validation.json','rb') as f: # 내가봤을때는 여기만 수정하면 될거 같은데. 바운딩 박스로다가. 
    train_json = json.load(f)
    
with open('/Data1/home/hshong/workspace/STIGPN/S-Else/dataset/bounding_box/bouding_box_smthsmth_dict.json','rb') as f: # 내가봤을때는 여기만 수정하면 될거 같은데. 바운딩 박스로다가. 
    bounding_box_annotations = json.load(f)
# 여기는 바운딩박스에서 train.json만 골라서 만드는거 그리고 다시 bounding_box 내에서 traion 이랑 val로 split한거.

#id_list = list(bounding_box_annotations.keys())

id_list = []
for i in range(0,len(train_json)):
    id_list.append(train_json[i]['id'])

json_dict = []
box_dict = {}

#key value를 같이 넣어야지, 
for list in id_list:
    box_dict[list] = bounding_box_annotations[list]


for i in id_list:
  for j in range(0,len(train_json)):
   if train_json[j]['id'] == i:
     json_dict.append(train_json[j])


with open("box_val.json", 'w') as outfile:
    json.dump(box_dict, outfile)

'''


#with open('/Data1/home/hshong/workspace/STIGPN/box_train.json','rb') as f:
 #   data_jsons2 = json.load(f)

#with open('/Data1/home/hshong/workspace/STIGPN/box_val.json','rb') as f:
 #   data_jsons3 = json.load(f)
    

#print(len(data_jsons2))
#print(len(data_jsons3))

import pickle
with open('/Data1/home/hshong/workspace/STIGPN/S-Else/dataset/temp_pkl/val_temp_data.pkl', 'rb') as f:
    data = pickle.load(f)
    

print(data['labels'])




