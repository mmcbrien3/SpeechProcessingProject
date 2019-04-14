from FileHandler import FileHandler
from FeatureExtractor import FeatureExtractor
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys, os
from matplotlib import pyplot as plt
types = ["pure_speech", "music"]
#get all file names
#create list of file names for training and testing

#for each file, record features and store
#train classifier
#test classifier
#store classifier

def get_data_from_folder(folder, classification, train_p, total_amount, valid_ext):
    max_train = round(total_amount * train_p)
    max_test = round(total_amount * (1-train_p))

    x_train = np.empty((max_train, extractor.tot_features), float)
    y_train = np.empty((max_train, 1), float)
    x_test = np.empty((max_test, extractor.tot_features), float)
    y_test = np.empty((max_test, 1), float)
    print("Begin %s" % (classification))
    fhandler = FileHandler(folder, training_percent=train_p)
    fhandler.set_file_extensions(valid_ext)
    fhandler.create_all_file_list()
    fhandler.split_train_test()
    count = 0
    print("Get %d training pts" % round(max_train))
    for tf in fhandler.train_files:
        cur_clips = fhandler.file_to_clips(tf)
        for clip in cur_clips:
            extractor.set_clip(clip)
            x_train[count][:] = extractor.extract()
            y_train[count] = classification
            count = count + 1
            if count % 100 == 0:
                print(count)
            if count == max_train:
                break
        if count == max_train:
            break
    print("Collected %d of %d requested %s samples" % (count, max_train, classification))
    x_train = x_train[0:count, :]
    y_train = y_train[0:count, :]
    print("Get %d testing" % (max_test))
    count = 0
    for tf in fhandler.test_files:
        cur_clips = fhandler.file_to_clips(tf)
        for clip in cur_clips:
            extractor.set_clip(clip)
            x_test[count][:] = extractor.extract()
            y_test[count] = classification
            count = count + 1
            if count % 100 == 0:
                print(count)
            if count == max_test:
                break
        if count == max_test:
            break
    x_test = x_test[0:count, :]
    y_test = y_test[0:count, :]

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":

    extractor = FeatureExtractor()

    tts = 0.8
    max_data = 1000
    num_classes = 3
    all_x_train = [""] * num_classes
    all_y_train = [""] * num_classes
    all_x_test = [""] * num_classes
    all_y_test = [""] * num_classes
    
    all_x_train[0], all_y_train[0], all_x_test[0], all_y_test[0] = get_data_from_folder(".\\SpeechFolder\\PureSpeech", 0,
                                                            tts, max_data, (".wav"))

    all_x_train[1], all_y_train[1], all_x_test[1], all_y_test[1] = get_data_from_folder(".\\SpeechFolder\\Music", 1,
                                                            tts, max_data, (".wav"))

    all_x_train[2], all_y_train[2], all_x_test[2], all_y_test[2] = get_data_from_folder(".\\SpeechFolder\\Noise", 2,
                                                                    tts, max_data, (".wav"))
    x_train = np.empty((0, extractor.tot_features), float)
    y_train = np.empty((0, 1), float)
    x_test = np.empty((0, extractor.tot_features), float)
    y_test = np.empty((0, 1), float)
    for i in range(len(all_x_train)):
        x_train = np.append(x_train, all_x_train[i], axis=0)
        y_train = np.append(y_train, all_y_train[i], axis=0)
        x_test = np.append(x_test, all_x_test[i], axis=0)
        y_test = np.append(y_test, all_y_test[i], axis=0)

    print(x_train)
    print(y_test)
    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(x_train, y_train)

    fhT = FileHandler(".\\SpeechFolder\\TEST")
    fhT.set_file_extensions((".wav"))
    fhT.create_all_file_list()
    fhT.split_train_test()

    pred = rf.predict(x_test)

    print(pred)
    print(y_test.flatten())
    sc = rf.score(x_test, y_test.flatten())
    print(sc)


