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
        while(1):
            data = self.recorder.buffer.get()
            result = self.detection.forward(data)
            qs = self.recorder.buffer.qsize()
            sys.stdout.write(' '*100+'\r')
            sys.stdout.write('quesize: {:d}\t'.format(qs))
            if result[0].shape[0] == 0:
                sys.stdout.write('Silent\r')
            else:
                for i in result[0]:
                    sys.stdout.write(parameter.id_name[i]+' ')
                sys.stdout.write('\r')
            sys.stdout.flush()

class Detection():
    def __init__(self):
        config = runner.Config()
        self.net = runner.prepare(config, stage='evaluate', noneed_data=True)
        self.data = np.zeros((1,1,int(config.segment_length*1000./parameter.hop_length),40))
        self.extractor = Extractor.MelExtractor(n_fft=2048, fmin=0., fmax=22050., htk=True,logarithmic=True)
        self.history_half_segments = np.zeros(int(parameter.hop_length*44100/1000)) 

    def forward(self, data):
        # data []
        self.data = np.roll(self.data, -2, axis=2)
        context = np.concatenate((self.history_half_segments, data))
        feat = self.extractor.extract(context).T[1:-1]
        self.data[0,0,-2:,:] = feat
        output = self.net(torch.from_numpy(self.data).float().to(parameter.device))
        output = output.cpu().detach().numpy()[:,-1,:]
        predict = (output>0.5).astype(np.float).squeeze()
        self.history_half_segments = data[-parameter.hop_length*44100/1000:]
        return predict.nonzero() 
        
        
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
