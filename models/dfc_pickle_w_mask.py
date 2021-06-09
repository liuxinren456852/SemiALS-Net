import os
import pickle
import sys
import numpy as np

def convert_label(label):
    '''
    ignored_labels = np.array([0])
        idx_label = [2,5,6,9,17]
    '''
    label[label==2]=1
    label[label==5]=2
    label[label==6]=3
    label[label==9]=4
    label[label==17]=5
    
    return label

class DFCDataset():
    def __init__(self, root='./data/', npoints=4096, split='train', log_weighting=True, extra_features=[]):  # 4096 in default
        self.npoints = npoints
        self.root = root
        self.split = split
            
        if split=='train' or split=='train_val':
            train_f = open('./data/train_1599.pickle', 'rb')
            data, labels, feats = pickle.load(train_f, encoding='iso-8859-1') #,encoding='iso-8859-1'
            train_f.close()
        elif split=='test':
            train_f = open('./data/test_160.pickle', 'rb')
            data, labels, feats = pickle.load(train_f, encoding='iso-8859-1') #,encoding='iso-8859-1'
            train_f.close()
        else:
            assert 'data not exist'
        
        self.data = data
        self.labels = labels
        self.feats = feats
        self.columns = np.array([0,1,2]+extra_features)
        
        self.M = 6 #with unknown class
        self.compressed_label_map = {0:0, 2:1, 5:2, 6:3, 9:4, 17:5}
        self.decompress_label_map = {0:0, 1:2, 2:5, 3:6, 4:9, 5:17}

        if split=='train' or split=='train_val':
            label_w = convert_label(np.concatenate(labels))
            label_w = label_w[label_w>0]
            tmp,_ = np.histogram(label_w, range(1, self.M+1))
            labelweights = tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            labelweights = 1/np.log(1.1+labelweights)
            self.labelweights = np.insert(labelweights, 0, 0)
            print(self.labelweights)
        else:
            self.labelweights = np.ones(self.M, dtype='float32')
            print(self.labelweights)
        
    def __getitem__(self, index):
        point_set = self.data[index]
        labels = self.labels[index]
        feats = self.feats[index]
        n = point_set.shape[0]
        
        point_set = point_set-point_set.mean(0)
        point_set = point_set/point_set.max(0)
        feats = feats - feats.mean(0)
        feats = feats/feats.max(0)
        
        if self.npoints < n:
            idxs = np.random.choice(n,self.npoints,replace=False)
        elif self.npoints == n:
            idxs = np.arange(self.npoints)
        else:
            idxs = np.random.choice(n,self.npoints,replace=True)
        
        point_set = np.concatenate([point_set[idxs,:], feats[idxs,:]], 1)
        semantic_seg = convert_label(labels[idxs,])
        sample_weight = self.labelweights[semantic_seg]
        
        sup_mask = np.ones(self.npoints, dtype='int32')
#         for i in range(self.npoints):
#             sup_mask[i] = labels[ixs[i],1]
            
        return point_set, semantic_seg, sample_weight, sup_mask

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    d = DFCDataset(root = './data/dfc_v4b', extra_features=[3,4])
    point_set, semantic_seg, sample_weight = d[0]
    print(point_set)
    print(semantic_seg)
    print(sample_weight)
    print("Scale:"+str(d.scale))
    print("Mapping: "+str(d.decompress_label_map))
    print("Weights: "+str(d.labelweights))
    print("Counts: "+str(d.cls_hist))
    tmp = np.array(list(d.cls_hist.values()),dtype=float)
    print("Frequency: "+str(tmp/np.sum(tmp)))
    print("Length: "+str(len(d.data)))
    exit()