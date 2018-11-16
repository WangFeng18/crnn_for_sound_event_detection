import os
import numpy as np
import parameter
import cPickle
import torch
import torch.utils.data.dataset as dataset

class EventStamp:
    def __init__(self, meta):
        metafile, featfile = meta
        with open(metafile, 'r') as f:
            self.lines = f.readlines()
        feat_meta = cPickle.load(open(featfile, 'rb'))
        self.N = feat_meta['stat'][0]['N']
        self.feat = feat_meta['feat'][0]
        self.stamps = np.zeros((self.N, parameter.n_classes))
        for line in self.lines:
            start, end, event = line.split()
            start = float(start)
            end = float(end)
            self.set_true(start, end, event)
        self.feats, self.stamps = self.segments()

    def set_true(self, start_time, end_time, event):
        start_frame = int(start_time*1000./parameter.hop_length)
        end_frame = int(end_time*1000./parameter.hop_length)
        id = parameter.name_id[event]
        for i in range(start_frame, min(end_frame+1, self.N)):
            self.stamps[i][id] = 1
    def segments(self):
        seg_frames = parameter.segment_length*1000/parameter.hop_length
        start = 0  
        segs = []
        feats = []
        while start < self.N:
            if start + seg_frames < self.N:
                segs.append(self.stamps[start:start+seg_frames,:])
                feats.append(self.feat[start:start+seg_frames,:])
            else:
                segs.append(self.stamps[self.N-seg_frames:self.N,:])
                feats.append(self.feat[self.N-seg_frames:self.N,:])
            start += seg_frames
        return feats, segs
        

class data(dataset.Dataset):
    def __init__(self, meta=parameter.train):
        self.data = []
        self.stamp = []
        for i, single_meta in enumerate(meta):
            es = EventStamp(single_meta)
            self.data.extend(es.feat)
            self.stamp.extend(es.stamps)

    def __getitem__(self, index):
        return (self.data[i], self.stamp[i])

    def __len__(self):
        return len(self.data)

d = data()

            
