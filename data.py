import os
import numpy as np
import parameter
import cPickle
import torch
import torch.utils.data.dataset as dataset
import sed_tools.annotation as sedannot

# two reciprocal function:
'''
    annot_file           annot_file
        |                    |
        |                    |
      label             concat_label
        |                    |
        |                    |
    seg_label          esti_seg_label
        |                    |
        |------Inference-----|
'''
def parse_mix(mix_id, seg_length, stride, padding):
    metafile = os.path.join(parameter.meta_dir, parameter.meta_template.format(mix_id))
    featfile = os.path.join(parameter.feature_dir, parameter.feat_template.format(mix_id))
    with open(metafile, 'r') as f: lines = f.readlines()
    feat_meta = cPickle.load(open(featfile, 'rb'))
    N = feat_meta['stat'][0]['N']
    feat = feat_meta['feat'][0]
    
    label = sedannot.annot2label(lines, N, parameter)
    feats, labels = sedannot.segments(feat, label, parameter, N, seg_length=seg_length, stride=stride, padding=padding)
    ids = [mix_id] * len(feats)
    lengths = [N] * len(feats)
    return feats, labels, ids, lengths


class data(dataset.Dataset):
    def __init__(self, stage='train', config=None):
        mix_ids = getattr(parameter, stage)
        self.config = config
        self.data = []
        self.labels = []
        self.ids = []
        self.lengths = []
        stride = config.stride if stage=='train' else -1
        seg_length = config.segment_length
        for i, mix_id in enumerate(mix_ids):
            padding = False if stage=='train' else True
            feats, labels, ids, lengths = parse_mix(mix_id, seg_length, stride, padding)
            self.data.extend(feats)
            self.labels.extend(labels)
            self.ids.extend(ids)
            self.lengths.extend(lengths)

    def __getitem__(self, i):
        return (torch.from_numpy(self.data[i]).unsqueeze(0), torch.from_numpy(self.labels[i]),
                self.ids[i], self.lengths[i])

    def __len__(self):
        return len(self.data)

