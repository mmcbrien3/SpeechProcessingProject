import os, random
import numpy as np
from scipy import signal
import sounddevice
from scipy.io import wavfile


class FileHandler(object):

    def __init__(self, root_dir, training_percent=1):
        self.root = root_dir
        self.extensions = ()
        self.all_files = []
        self.train_files = []
        self.test_files = []
        self.train_percent = training_percent
        self.sr = 8000

    def set_file_extensions(self, ext):
        self.extensions = ext

    def check_file_extension(self, f):
        return f.lower().endswith(self.extensions)

    def create_all_file_list(self):
        for r, subdirs, files in os.walk(self.root):
            for c_file in files:
                if self.check_file_extension(c_file):
                    self.all_files.append(os.path.join(r, c_file))

    def split_train_test(self):
        random.shuffle(self.all_files)
        total_train = round(self.train_percent * len(self.all_files))
        self.train_files = self.all_files[0:total_train]
        self.test_files = self.all_files[total_train:]

    def get_type(self, f):
        if "purespeech" in f.lower():
            return 1
        elif "music" in f.lower():
            return 2
        else:
            return -1

    def resample(self, audiofile, new_fs):
        fs, raw = wavfile.read(audiofile)
        raw = raw / (2**15 - 1)
        num_samples = round(new_fs/fs*len(raw))
        raw = signal.resample(raw,num=num_samples)
        return raw

    def file_to_clips(self, audiofile):
        raw = self.resample(audiofile, self.sr)
        num_clips = int(np.floor(len(raw)/self.sr))
        clips = np.zeros((num_clips, self.sr))
        if len(raw.shape) > 1:
            raw = raw[:, 0]
        for i in range(num_clips):
            clips[i][:] = raw[i*self.sr:(i+1)*self.sr]
        return clips

