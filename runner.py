import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data.dataloader as dl
from data import data
from sed_tools.annotation import label2annot
import os
import numpy as np
from tensorboardX import SummaryWriter
import time
import fire
import torchvision.transforms as transforms
import parameter
from utils import progress_bar
from net import Net
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# difference with parameter and config:
# parameter store some constant suchas path, data format, n_classes etc.
# config is used to test the best model, so it stores variable hyparameter or some values needed to be adjust.
class Config:
    # training parameter
    def __init__(self, max_epoch = 1000,lr = 0.00001, batch_size = 64, stride = -1, segment_length = 20.,net = 'NETA'):
        self.max_epoch = max_epoch 
        self.lr = lr
        self.batch_size = batch_size

        self.stride = stride
        self.segment_length = segment_length
        self.net = net
        self.name = self._getname()

    def _getname(self):
        return '_'.join([self.net, 'sl'+str(self.segment_length), parameter.feature_type])

    

def prepare(config, stage, load_model=False, noneed_data=False):
    net = Net(config)
    #net = nn.DataParallel(net)
    if load_model or stage != 'train': 
        net.load_state_dict(torch.load(os.path.join('model', '{}.pth'.format(config.name))))
    if parameter.use_gpu:
        net.cuda()
        cudnn.benchmark = True
    if noneed_data:return net
    if stage=='train':
        dataloader = dl.DataLoader(data(stage='train', config=config), \
                batch_size=config.batch_size, shuffle=True, num_workers=4)
    elif stage=='evaluate':
        dataloader = dl.DataLoader(data(stage='evaluate', config=config), batch_size=100, num_workers=4)
    else:
        dataloader = dl.DataLoader(data(stage='test', config=config), batch_size=100, num_workers=4)
    return dataloader, net

def train(net, dataloader, config):
    #criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    #optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    # The initial lr will be decayed by gamma every step_size epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    writer = SummaryWriter()
    #dummy_input = torch.Tensor(32, 3, 32, 32)
    #writer.add_graph(net, (dummy_input,))
    for i_epoch in range(config.max_epoch):
        scheduler.step()
        epoch_loss = 0.
        epoch_start = time.time()
        n_total = 0.
        n_correct = 0.
        print('Epoch:{}'.format(i_epoch))
        for i_batch, data in enumerate(dataloader):
            input = data[0].float()
            label = data[1].float()
            if parameter.use_gpu:
                input = input.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, label)
            epoch_loss += loss.item()

            #_, predicted = output.max(1)
            #n_total += label.size(0)
            #n_correct += predicted.eq(label).sum().item()

            loss.backward()
            optimizer.step()
            progress_bar(i_batch, len(dataloader), 'L:{:.4f}'.\
                    format(epoch_loss/(i_batch+1), ))
        epoch_end = time.time()
        writer.add_scalar('epoch_loss', epoch_loss/(i_batch+1), i_epoch)
        writer.add_scalar('epoch_time', epoch_end-epoch_start, i_epoch)
        #test(net, test_dataloader, False)
        torch.save(net.state_dict(), os.path.join('model', '{}.pth'.format(config.name)))

def test(net, dataloader):
    net.eval()
    n_correct = 0.
    n_total = 0.
    epoch_loss = 0.
    criterion = nn.BCELoss()
    last_id = -1
    last_lenth = -1
    total_annot = [None]*100
    annot = None
    for i_batch, data in enumerate(dataloader):
        input = data[0].float()
        label = data[1].float()
        if parameter.use_gpu:
            input = input.cuda()
            label = label.cuda()
        id = data[2]
        length = data[3]
        
        output = net(input)
        loss = criterion(output, label)

        output = output.cpu().detach().numpy()
        predict = (output>=0.5).astype(np.float)

        for i_sample in range(id.shape[0]):
            current_id = id[i_sample]
            if current_id != last_id:
                if annot is not None:
                    total_annot[last_id] = annot[:last_length]
                annot = predict[i_sample]
            else:
                annot = np.concatenate((annot, predict[i_sample]), axis=0)
            last_id = current_id
            last_length = length[i_sample]
        epoch_loss += loss.item()
        
        progress_bar(i_batch, len(dataloader), 'L:{:.4f}'.format(epoch_loss/(i_batch+1)))
    total_annot[last_id] = annot[:last_length]
    for i in range(100):
        if total_annot[i] is not None:
            print i
            lines = label2annot(total_annot[i], parameter)
            with open(os.path.join(parameter.estimate_dir, parameter.meta_template.format(i)), 'w+') as f:
                f.writelines(lines)
    net.train()



def firefunc(config, istrain=True, load_model=False):
    dataloader, net = prepare(config, stage=('train' if istrain else 'evaluate'), load_model=load_model)
    if istrain:train(net, dataloader, config)
    else: 
        with torch.no_grad():
            test(net, dataloader)

if __name__ == '__main__':
    config = Config(segment_length=5.)
    fire.Fire(lambda istrain=True, load_model=False:firefunc(config, istrain, load_model))
