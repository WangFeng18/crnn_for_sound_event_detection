import os
import numpy as np
import parameter
import cPickle
import torch
import torch.utils.data.dataset as dataset
import sed_tools.annotation as sedannot
import dcase_util
import dcase_util.features.features as extractor

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
    if parameter.feature_type == 'O':
        feat_meta = cPickle.load(open(featfile, 'rb'))
        feat = feat_meta['feat'][0] # Nx40
        mean = np.expand_dims(feat_meta['stat'][0]['mean'], axis=0)
        std = np.expand_dims(feat_meta['stat'][0]['std'], axis=0)
        feat = (feat-mean)/(std+0.0000001)
    else:# X
        if not os.path.exists(featfile):
            print('generating features of {}'.format(featfile))
            audio_path = os.path.join(parameter.audio_dir, parameter.audio_template.format(mix_id))
            audio_container = dcase_util.containers.AudioContainer().load(filename=audio_path).mixdown()
            a = extractor.MelExtractor(n_fft=2048, fmin=0., fmax=22050., htk=True,logarithmic=True)
            feat = a.extract(audio_container).T
            np.save(featfile, feat)
        else:
            feat = np.load(featfile)

    N = feat.shape[0]
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

