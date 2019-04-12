from FileHandler import FileHandler
from FeatureExtractor import FeatureExtractor
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
from matplotlib import pyplot as plt
types = ["silence", "music", "pure speech", "background noise", "impure speech"]
#get all file names
#create list of file names for training and testing

#for each file, record features and store
#train classifier
#test classifier
#store classifier

if __name__ == "__main__":

    extractor = FeatureExtractor()
    x_train = np.empty((2000, extractor.tot_features), float)
    y_train = np.empty((2000, 1), float)
    x_test = np.empty((0, extractor.tot_features), float)
    y_test = np.empty((0, 1), str)

    ## HANDLE PURE SPEECH DATA ##
    print("Begin Pure Speech")
    fhPS = FileHandler(".\\SpeechFolder\\PureSpeech")
    fhPS.set_file_extensions((".wav"))
    fhPS.create_all_file_list()
    fhPS.split_train_test()
    max_data = 1000
    count = 0

    for tf in fhPS.train_files:
        cur_clips = fhPS.file_to_clips(tf)
        for clip in cur_clips:
            extractor.set_clip(clip)
            x_train[count][:] = extractor.extract()
            y_train[count] = fhPS.get_type(tf)
            count = count + 1
            if count % 10 == 0:
                print(count)
            if count == max_data:
                break
        if count == max_data:
            break

    ## HANDLE MUSIC DATA ##
    print("Begin Music")
    fhM = FileHandler(".\\SpeechFolder\\MUSIC")
    fhM.set_file_extensions((".wav"))
    fhM.create_all_file_list()
    fhM.split_train_test()

    for tf in fhM.train_files:
        cur_clips = fhM.file_to_clips(tf)
        for clip in cur_clips:
            extractor.set_clip(clip)
            x_train[count][:] = extractor.extract()
            y_train[count] = fhPS.get_type(tf)
            count = count + 1
            if count % 10 == 0:
                print(count)
            if count == max_data*2:
                break
        if count == max_data*2:
            break

    print(x_train)
    print(y_train)
    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(x_train, y_train)

    fhT = FileHandler(".\\SpeechFolder\\TEST")
    fhT.set_file_extensions((".wav"))
    fhT.create_all_file_list()
    fhT.split_train_test()
    for tf in fhT.train_files:
        cur_clips = fhT.file_to_clips(tf)
        for clip in cur_clips:
            extractor.set_clip(clip)
            x_test = np.append(x_train, np.array([extractor.extract()]), axis=0)
            count = count + 1

    y_test = rf.predict(x_test)
    print(y_test)


