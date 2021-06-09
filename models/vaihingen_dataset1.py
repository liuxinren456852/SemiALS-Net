import os
import pickle
import sys
import numpy as np


class DFCDataset():
    def __init__(self, root, npoints=4096, split='train', log_weighting=True, extra_features=[]):  # 4096 in default
        self.npoints = npoints
        self.root = root
        self.split = split
        
        # Dataset size causes memory issues with numpy.save; used pickle instead
        #self.data = np.load(os.path.join(self.root, 'dfc_{}_dataset.npy'.format(split)))
        with open(os.path.join(self.root, 'dfc_{}_dataset.pickle'.format(split)),'rb') as f:
            self.data = pickle.load(f)
        with open(os.path.join(self.root, 'dfc_{}_metadata.pickle'.format(split)),'rb') as f:
            self.metadata = pickle.load(f)
        with open(os.path.join(self.root, 'dfc_{}_labels.pickle'.format(split)),'rb') as f:
            self.labels = pickle.load(f)
        
        self.log_weighting = log_weighting
        self.extra_features = extra_features
        self.columns = np.array([0,1,2]+extra_features)
        
        self.M = 9
        
        if split=='train':
            scaling_metadata = self.metadata   # always using dfc_train_metadata
        else:
            with open(os.path.join(self.root, 'dfc_train_metadata.pickle'),'rb') as f:
                scaling_metadata = pickle.load(f)
        
        scale = np.sqrt(scaling_metadata['variance'])
        scale[0:3] = np.sqrt(np.mean(np.square(scale[0:3])))
        self.scale = scale
        self.cls_hist = scaling_metadata['cls_hist']
        
        self.compressed_label_map = scaling_metadata['compressed_label_map']
        self.decompress_label_map = scaling_metadata['decompress_label_map']
        
        self.labelweights = np.zeros(self.M, dtype='float32')
        for key, ix in self.compressed_label_map.items():
            self.labelweights[ix] = self.cls_hist[key]

        if split=='train' or split=='train_val':
            self.labelweights = self.labelweights/np.sum(self.labelweights)
            self.class_freq = self.labelweights
            if self.log_weighting:  # do log function
                self.labelweights = 1/np.log(1.2+self.labelweights)   # modify a, to NA,1.0, in default: 1.2
#                 self.labelweights[0] += 5
                print('class balanced weights', self.labelweights) 
                #[5.4709616 2.6691446 2.6733463 5.3321414 5.133167  3.1006021 4.706809 4.2879157 3.0366802]
            else:
                self.labelweights = np.sqrt(1/self.labelweights)
        else:
            self.labelweights = np.ones(self.M, dtype='float32')
        

    def __getitem__(self, index):
        point_set = self.data[index]
        labels = self.labels[index]
        n = point_set.shape[0]
        
        if self.npoints < n:
            ixs = np.random.choice(n,self.npoints,replace=False)
        elif self.npoints == n:
            ixs = np.arange(self.npoints)
        else:
            ixs = np.random.choice(n,self.npoints,replace=True)
        
        tmp = point_set[ixs,:]
        point_set = tmp[:,self.columns] / self.scale[self.columns]
        semantic_seg = np.zeros(self.npoints, dtype='int32')
        for i in range(self.npoints):
            semantic_seg[i] = self.compressed_label_map[labels[ixs[i],0]]
        sample_weight = self.labelweights[semantic_seg]
        
        return point_set, semantic_seg, sample_weight

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