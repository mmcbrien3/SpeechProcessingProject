import soundfile
import numpy as np
import scipy as sp

class FeatureExtractor(object):

    tot_features = 4 * 2 + 12 * 2
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
        features.extend(self.get_mel_freq_cepstral_coeffs(subclips))
        features.extend(self.get_spectral_centroid(subclips))
        features.extend(self.get_spectral_flux(subclips))
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

    def get_spectral_centroid(self, subclips):
        scs = np.zeros(len(subclips))
        count = 0
        sr = 8000
        for sc in subclips:
            mags = np.abs(np.fft.rfft(sc))
            length = len(sc)
            freqs = np.abs(np.fft.fftfreq(length, 1.0 / sr)[:length // 2 + 1])
            scs[count] = np.sum(mags * freqs) / np.sum(mags)
            count += 1

        return [np.mean(scs), np.std(scs)]

    def get_mel_freq_cepstral_coeffs(self, subclips):
        mfccs = []
        for sc in subclips:
            si_k = np.fft.fft(sc, n=512)
            pi_k = (1/len(si_k)) * (np.abs(si_k) ** 2)
            pi_k = pi_k[:257]
            fil_banks = self.compute_filterbank()
            fil_energies = np.sum(pi_k * fil_banks, axis=1)
            log_energies = np.log10(fil_energies)
            cepstral_coeff = sp.fftpack.dct(log_energies, norm='ortho')
            mfcc = cepstral_coeff[:12]
            mfccs.append(mfcc)
        ms = np.mean(mfccs, axis=0)
        stds = np.std(mfccs, axis=0)
        return np.concatenate((ms, stds))

    def compute_filterbank(self, nfft=512, nfilt=26):
        lowmel = self.convert_freq_to_mel(0)
        highmel = self.convert_freq_to_mel(8000/2)

        melpoints = np.linspace(lowmel,highmel,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin_number = np.floor((nfft+1)*self.convert_mel_to_freq(melpoints)/8000)

        fbank = np.zeros([nfilt,nfft//2+1])
        for j in range(0,nfilt):
            for i in range(int(bin_number[j]), int(bin_number[j+1])):
                fbank[j,i] = (i - bin_number[j]) / (bin_number[j+1]-bin_number[j])
            for i in range(int(bin_number[j+1]), int(bin_number[j+2])):
                fbank[j,i] = (bin_number[j+2]-i) / (bin_number[j+2]-bin_number[j+1])
        return fbank

    def convert_freq_to_mel(self, frequency):
        return 2595 * np.log10(1+frequency/700.)

    def convert_mel_to_freq(self, mel):
        return 700 * np.exp((mel/1125.) - 1)

    def get_spectral_flux(self, subclips):
        sfs = np.zeros(len(subclips)-1)
        epsilon = 0.000000001
        prev_fft = np.abs(np.fft.fft(subclips[0]))
        for i in range(1, len(subclips)):
            cur_fft = np.abs(np.fft.fft(subclips[i]))
            sum_cur_fft = np.sum(cur_fft + epsilon)
            sum_prev_fft = np.sum(prev_fft + epsilon)
            sfs[i-1] = np.sum((cur_fft/sum_cur_fft - prev_fft/sum_prev_fft) ** 2)
            prev_fft = cur_fft

        return [np.mean(sfs), np.std(sfs)]

