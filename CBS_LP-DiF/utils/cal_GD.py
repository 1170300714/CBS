import os
import pickle
import numpy as np



def calculate_GD(data_list,begin_class,class_num):
    GD_dict = {x:[] for x in range(begin_class,begin_class+class_num)}
    feat_dict = {x:[] for x in range(begin_class,begin_class+class_num)}
    dim=512


    for i in range(len(data_list)):
        data = data_list[i]
    
        feat = data['feats'][0]
    
        label = data['label'][0]
        # print(label)

        feat_dict[label].append(feat)


    for key in feat_dict:
        feat_list = np.array(feat_dict[key])
        if feat_list.shape[0]<2:
            continue

        for j in range(dim):
            f_j = feat_list[:,j]
            mean = np.mean(f_j)
            std = np.std(f_j,ddof=1)
            GD_dict[key].append({'mean':mean,'std':std})

    return GD_dict



# task_number = 20

# base_class = 10
# per_task_class = 10

# dim = 512

# root_temp = 'extract_feature/cub200_wo_base_fully_task{}_feat'
# GD_dict = {x:[] for x in range(all_class)}
# for task_id in range(task_number):
#     print(task_id)
    
#     root= root_temp.format(task_id)
#     if task_id==0:
#         start=0
#         end = base_class
#     else:
#         start=base_class+(task_id-1)*per_task_class
#         end=start+per_task_class

#     feat_dict = {x:[] for x in range(start,end)}

#     average_list = []
#     for i in range(len(os.listdir(root))):
#         file_path = os.path.join(root,'{}.pkl'.format(i))
#         with open(file_path,'rb') as f:
#             data = pickle.load(f)
    
#         feat = data['feats'][0]
      
#         label = data['label'][0]
#         # print(label)



#         feat_dict[label].append(feat)

#     # root2 = '/home/huangzitong/projects/SHIP-main/gen_feat_task{}'.format(task_id)
#     # for i in range(len(os.listdir(root2))):
#     #     file_path = os.path.join(root2,'{}.pkl'.format(i))
#     #     with open(file_path,'rb') as f:
#     #         data = pickle.load(f)
#     #     feat = data['feats'][0]

#     #     label = int(data['label'])

#     #     # print(label)
#     #     if len(feat_dict[label]) <= 50:
#     #         feat_dict[label].append(feat)
#     for key in feat_dict:
#         feat_list = np.array(feat_dict[key])

#         # print(feat_list.shape)



#         # average_list.append(np.mean(feat_list,axis=0))

#         for j in range(dim):
#             f_j = feat_list[:,j]
#             mean = np.mean(f_j)
#             std = np.std(f_j,ddof=1)
#             GD_dict[key].append({'mean':mean,'std':std})

# print(GD_dict.keys())
# with open("cub200_wo_base_fully_GD.pkl".format(task_id),'wb') as f:
#     pickle.dump(GD_dict,f,-1)

# print(GD_dict)