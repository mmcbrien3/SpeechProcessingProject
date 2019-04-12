import soundfile
import numpy as np

class FeatureExtractor(object):

    tot_features = 2 * 2
    def __init__(self):
        pass

    def set_clip(self, clip):
        self.raw = clip

    def get_sub_clips(self):
        len_subclip = int(len(self.raw)/40)
        sc = np.zeros((40, len_subclip))
        for i in range(40):
            sc[i][:] = self.raw[i*len_subclip:(i+1)*len_subclip]
        return sc

    def extract(self):
        subclips = self.get_sub_clips()
        features = []
        features.extend(self.get_energy(subclips))
        features.extend(self.get_zero_crossing(subclips))
        return features

    def get_energy(self, subclips):
        energies = np.zeros(len(subclips))
        count = 0
        for sc in subclips:
            energies[count] = np.sum(np.square(sc))
            count = count + 1

        return [np.mean(energies), np.std(energies)]

    def get_zero_crossing(self, subclips):
        zcs = np.zeros(len(subclips))
        count = 0
        for sc in subclips:
            zcs[count] = np.size(np.where(np.diff(np.sign(sc)))[0])
            count = count + 1

        return [np.mean(zcs), np.std(zcs)]