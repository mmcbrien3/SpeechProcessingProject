from FileHandler import FileHandler
from FeatureExtractor import FeatureExtractor
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
class RealWorldTester(object):

    def __init__(self, fileloc, clf, ss=None):
        self.clf = clf
        self.standardizer = ss
        self.file_loc = fileloc
        self.file_handler = FileHandler(self.file_loc)
        self.file_handler.set_file_extensions((".wav"))
        self.file_handler.create_all_file_list()
        self.file_handler.split_train_test()
        self.extractor = FeatureExtractor()

    def classify_clip(self):
        y = []
        for tf in self.file_handler.train_files:
            cur_clips = self.file_handler.file_to_clips(tf, overlap=False)
            for clip in cur_clips:
                self.extractor.set_clip(clip)
                features = np.reshape(self.extractor.extract(), (1, -1))
                features = self.standardizer.transform(features)
                pred = self.clf.predict(features)
                y.append(pred[0])

        return y


    def plot_classification(self):
        classification = self.classify_clip()
        class_colors = ["#4286f4", "#ff3855", "#92ff5b", "#f4c60e", "#111111"]

        p_sp = []
        im_sp = []
        music = []
        noise = []
        sil = []
        count = 0
        for c in classification:
            if c == 'p':
                p_sp.append(count)
            elif c == 'i':
                im_sp.append(count)
            elif c == 'm':
                music.append(count)
            elif c == 'n':
                noise.append(count)
            elif c == 's':
                sil.append(count)
            count += 1
        print(sil)
        print(noise)
        print(music)
        print(im_sp)
        print(p_sp)
        plt.figure()
        plt.plot(sil, [0] * len(sil), 'ko')
        plt.plot(noise, [1] * len(noise), 'yo')
        plt.plot(music, [2] * len(music), 'ro')
        plt.plot(im_sp, [3] * len(im_sp), 'bo')
        plt.plot(p_sp, [4] * len(p_sp), 'go')
        plt.yticks([0, 1, 2, 3, 4], ['silence', 'noise', 'music', 'impure speech', 'pure speech'])
        plt.show()


if __name__ == "__main__":
    clf = load("./SVC_classifier.joblib",)
    ss = load(".//standardizer.joblib")
    rwt = RealWorldTester(".//SpeechFolder//TEST/Brief_Test", clf, ss)

    rwt.plot_classification()