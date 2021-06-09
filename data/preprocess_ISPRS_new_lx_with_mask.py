import numpy
import os
from os.path import join
import pickle
import numpy as np
import pdb

NUM_POINT=4096
NUM_CLASSES = 9
def prep_pset(block, label):
    # data64 = np.stack([pset.x, pset.y, pset.z, pset.i, pset.r], axis=1)
    n = len(block)

    if NUM_POINT < n:
        # np.random.seed(5)
        ixs = np.random.choice(n, NUM_POINT, replace=False)
    elif NUM_POINT == n:
        ixs = np.arange(NUM_POINT)
    else:
        # np.random.seed(5)
        ixs = np.random.choice(n, NUM_POINT, replace=True)

    return block[ixs, 3:]/SCALE[3:], block[ixs, :3] / SCALE[:3], label[ixs]

data_root = './block_pickle_dense0.2/'
train_d = open(join(data_root, 'dfc_train_val_dataset.pickle'), 'rb')
train_dataset = pickle.load(train_d, encoding='bytes') # list:398 each member have point n*5


train_m= open(join(data_root, 'dfc_train_val_metadata.pickle'), 'rb')
train_metadata = pickle.load(train_m)
SCALE = np.sqrt(train_metadata['variance'])
SCALE[0:3] = np.sqrt(np.mean(np.square(SCALE[0:3])))

LABEL_MAP = train_metadata['decompress_label_map']

train_l= open(join(data_root, 'dfc_train_val_labels.pickle'), 'rb')
train_label_ori = pickle.load(train_l)

train_xyz=[]
train_feats=[]
train_label=[]
for block,label in zip(train_dataset,train_label_ori):
    feature, xyz, label = prep_pset(block,label)
    train_xyz.append(xyz)
    train_feats.append(feature)
    train_label.append(label)


test_d = open(join(data_root, 'dfc_test_dataset.pickle'), 'rb')
test_dataset = pickle.load(test_d, encoding='bytes') # list:398 each member have point n*5


test_l= open(join(data_root, 'dfc_test_labels.pickle'), 'rb')
test_label_ori = pickle.load(test_l)

test_xyz=[]
test_feats=[]
test_label=[]

for block, label in zip(test_dataset,test_label_ori):
    feature, xyz, label = prep_pset(block,label)
    test_xyz.append(xyz)
    test_feats.append(feature)
    test_label.append(label)

print('train length, test length', len(train_xyz), len(test_xyz))
all_xyz = train_xyz + test_xyz
all_label = train_label + test_label
all_feats = train_feats + test_feats


f=open(join(data_root, "sampled_train_data.pkl"),'wb')
pickle.dump([train_xyz,train_label,train_feats],f)
f.close()

f=open(join(data_root, "sampled_test_data.pkl"),'wb')
pickle.dump([test_xyz,test_label,test_feats],f)
f.close()

# cnt=1000
# idx=[]
# ratio=0.1
# for i in range(len(all_xyz)):
#     np.random.seed(cnt)  # assure different randoms
#     xyz=all_xyz[i]#n*3
#     label=all_label[i]#n*1
#     tmp,_=np.histogram(label,bins=9,range=(0,8))
#     each_index=[]
#     tmp2=np.asarray(np.ceil(tmp*ratio),dtype=int)
#     for i in range(0, 9):
#         ori_index=np.where(label==i)[0]
#         if tmp2[i]==0:
#             continue
#         index=np.random.choice(ori_index,tmp2[i]).tolist()
#         each_index=each_index+index
#     idx.append(each_index)
#     cnt+=1
# f=open(join(data_root, "sample_0.1_label.pkl"),'wb')
# pickle.dump(idx,f)
# f.close()


# labelweights = np.zeros(9)
# tmp,_ = np.histogram(train_label,range(10))
# print(tmp)
# labelweights = tmp
# labelweights = labelweights.astype(np.float32)
# labelweights = labelweights/np.sum(labelweights)
# print(labelweights)
# labelweights = 1/np.log(1.2+labelweights)
# print(labelweights)
