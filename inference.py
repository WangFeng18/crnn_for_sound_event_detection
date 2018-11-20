import threading
import logging
from pyaudio import PyAudio,paInt16,paFloat32
import numpy as np
import queue
import time
import runner
import parameter
import dcase_util
import dcase_util.features.features as Extractor
import torch
import sys

class RealTimeDetection():
    def __init__(self):
        self.detection = Detection()
        self.recorder = Recorder()
    def run(self):
        self.recorder.setDaemon(True)
        self.recorder.start()
        print('Detected Event:')
        while(1):
            data = self.recorder.buffer.get()
            result, exfeat_t, forward_t = self.detection.forward(data)
            qs = self.recorder.buffer.qsize()
            sys.stdout.write(' '*120+'\r')
            #sys.stdout.write('quesize: {:d}\t'.format(qs))
            if result[0].shape[0] == 0:
                sys.stdout.write('Silent ')
            else:
                for i in result[0]:
                    sys.stdout.write(parameter.id_name[i]+' ')
            sys.stdout.write('Total Time/Frame(40ms):{:.4f}ms, (Feature Extraction costs:{:.4f}ms, Forward costs:{:.4f}ms)'.format((exfeat_t+forward_t)*1000, exfeat_t*1000, forward_t*1000))
            sys.stdout.write('\r')
            sys.stdout.flush()

class Detection():
    def __init__(self):
        config = runner.Config(segment_length=20.)
        self.net = runner.prepare(config, stage='evaluate', noneed_data=True)
        self.data = np.zeros((1,1,int(config.segment_length*1000./parameter.hop_length),40))
        self.extractor = Extractor.MelExtractor(n_fft=2048, fmin=0., fmax=22050., htk=True,logarithmic=True)
        self.history_half_segments = np.zeros(int(parameter.hop_length*44100/1000)) 

    def forward(self, data):
        # data []
        st = time.time()
        self.data = np.roll(self.data, -2, axis=2)
        context = np.concatenate((self.history_half_segments, data))
        feat = self.extractor.extract(context).T[1:-1]
        self.data[0,0,-2:,:] = feat
        se = time.time()
        exfeat_t = se - st

        input = torch.from_numpy(self.data).float()
        if parameter.use_gpu:
            input = input.cuda()
        output = self.net(input)
        output = output.cpu().detach().numpy()[:,-1,:]
        predict = (output>0.5).astype(np.float).squeeze()
        self.history_half_segments = data[-parameter.hop_length*44100/1000:]
        st = time.time()
        forward_t = st-se
        return predict.nonzero(), exfeat_t, forward_t
        
        
class Recorder(threading.Thread):
    def __init__(self,
            time = None, #How much time to the end
            sr = 44100, #Sample rate
            batch_num  = 1764, #Batch size (how much data for a single fetch)
            frames_per_buffer = 1764
            ):
        threading.Thread.__init__(self)
        self.time = time
        self.sr = sr
        self.batch_num = batch_num
        #self.data_alter = threading.Lock()
        self.frames_per_buffer = frames_per_buffer
        #self.logger = logging.getLogger(__name__ + '.CoreRecorder')
        self.buffer = queue.Queue()
        self.start_time = None
        #self.__running = threading.Event()
        #self.__running.set()

    def run(self):
        #self.logger.debug("Start to recording...")
        #self.logger.debug("  Time = %s"%self.time)
        #self.logger.debug("  Sample Rate = %s"%self.sr)
        self.start_time = time.time()
        pa=PyAudio()
        stream=pa.open(format = paFloat32,channels=1, rate=self.sr,input=True, frames_per_buffer=self.frames_per_buffer)
        my_buf=[]
        count=0
        if self.time is None:
            total_count = 1e10
        else:
            total_count = self.time * self.sr / self.batch_num
        while count < total_count:
            datawav = stream.read(self.batch_num)
            datause = np.fromstring(datawav,dtype = np.float)
            self.buffer.put(datause)
            count+=1
        stream.close()

    def save_wave_file(self,filename,data):
        wf=wave.open(filename,'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.sr)
        wf.writeframes(b"".join(data))
        wf.close()

    def stop(self):
        self.__running.clear()

if __name__ == '__main__':
    RealTimeDetection().run()
