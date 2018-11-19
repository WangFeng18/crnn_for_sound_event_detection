import numpy as np

def label2annot(label, para):
    #label b x n x n_cls or t x n_cls
    if len(label.shape) == 2:
        label = np.expand_dims(label, axis=0)

    # padding zeros along the time axis
    # for convinient latter calculation
    label = np.concatenate((label, np.zeros((label.shape[0], 1, label.shape[2]))), axis=1)
    batch_annot = []
    for i_batch in range(label.shape[0]):
        annot = []
        for i_event in range(label.shape[2]):
            isstart = 0
            lasting_steps = 0
            for i_step in range(label.shape[1]):
                current = label[i_batch][i_step][i_event] 
                if current == 1 and isstart == 0:
                    isstart = 1
                    lasting_steps += 1
                if current == 1 and isstart == 1:
                    lasting_steps += 1
                if current == 0 and isstart == 1:
                    isstart = 0
                    start_time = (i_step - lasting_steps)*para.hop_length/1000.
                    end_time = (i_step - 1)*para.hop_length/1000.
                    annot.append('{}\t{}\t{}\n'.format(str(start_time), str(end_time), para.id_name[i_event]))
                    lasting_steps = 0
        batch_annot.append(annot)
    if i_batch == 0: return batch_annot[0]
    return batch_annot

def annot2label(lines, N, para):
    stamps = np.zeros((N, para.n_classes),dtype=np.float)
    def set_true(start_time, end_time, event):
        start_frame = int(start_time*1000./para.hop_length)
        end_frame = int(end_time*1000./para.hop_length)
        id = para.name_id[event]
        for i in range(start_frame, min(end_frame+1, N)):
            stamps[i][id] = 1
    for line in lines:
        start, end, event = line.split()
        start = float(start)
        end = float(end)
        set_true(start, end, event)
    return stamps

def segments(feature, label, para, N, seg_length=20., stride=-1,  padding=False):
    seg_frames = int(seg_length*1000/para.hop_length)
    if stride == -1: stride = seg_frames
    start, segs, feats = 0, [], [] 
    while start < N:
        if start + seg_frames <= N:
            segs.append(label[start:start+seg_frames,:])
            feats.append(feature[start:start+seg_frames,:])
            if start+seg_frames == N:break
        else:
            if padding:
                segs.append(np.concatenate((label[start:N,:], \
                            np.zeros((seg_frames-N+start,para.n_classes))), axis=0))
                feats.append(np.concatenate((feature[start:N,:], \
                            np.zeros((seg_frames-N+start,feature.shape[1]))), axis=0))
            else:
                segs.append(label[N-seg_frames:N,:])
                feats.append(feature[N-seg_frames:N,:])
            break
        start += stride
    return feats, segs

