from FileHandler import FileHandler
from FeatureExtractor import FeatureExtractor
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from RealWorldTester import RealWorldTester
from joblib import dump, load
import sys, os
from matplotlib import pyplot as plt
types = ["pure_speech", "music", "noise", "impure_speech", "silence"]
#get all file names
#create list of file names for training and testing

#for each file, record features and store
#train classifier
#test classifier
#store classifier

def get_data_from_folder(folder, classification, train_p, total_amount, valid_ext, extractor):
    max_train = round(total_amount * train_p)
    max_test = round(total_amount * (1-train_p))

    x_train = np.empty((max_train, extractor.tot_features), float)
    y_train = np.empty((max_train, 1), str)
    x_test = np.empty((max_test, extractor.tot_features), float)
    y_test = np.empty((max_test, 1), str)
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

def standardize_data(x_train, x_test):
    ss = preprocessing.StandardScaler().fit(x_train)
    x_train = ss.transform(x_train)
    x_test = ss.transform(x_test)
    dump(ss, "standardizer.joblib")
    return x_train, x_test

def test_classifier(clf, x_test, y_test):
    pred = clf.predict(x_test)
    sc = clf.score(x_test, y_test.flatten())
    cf = confusion_matrix(y_test, pred, labels=['p', 'm', 'n', 'i', 's'], )
    cf = np.asarray(cf, dtype="float32")
    for i in range(len(types)):
        cf[i, :] = cf[i, :] / np.sum(cf, axis=1)[i]
    print("Results for the %s classifier" % type(clf).__name__)

    print("Accuracy: %f" % sc)
    print(cf)



def extract_and_save():
    extractor = FeatureExtractor()

    tts = 0.8
    max_data = 1000
    num_classes = 5
    all_x_train = [""] * num_classes
    all_y_train = [""] * num_classes
    all_x_test = [""] * num_classes
    all_y_test = [""] * num_classes
    
    all_x_train[0], all_y_train[0], all_x_test[0], all_y_test[0] = get_data_from_folder(".\\SpeechFolder\\ImpureSpeech", types[3],
                                                                                        tts, max_data, (".wav"), extractor)

    all_x_train[1], all_y_train[1], all_x_test[1], all_y_test[1] = get_data_from_folder(".\\SpeechFolder\\Music", types[1],
                                                                                        tts, max_data, (".wav"), extractor)

    all_x_train[2], all_y_train[2], all_x_test[2], all_y_test[2] = get_data_from_folder(".\\SpeechFolder\\Noise", types[2],
                                                                                        tts, max_data, (".wav"), extractor)

    all_x_train[3], all_y_train[3], all_x_test[3], all_y_test[3] = get_data_from_folder(".\\SpeechFolder\\PureSpeech", types[0],
                                                                                        tts, max_data, (".wav"), extractor)

    all_x_train[4], all_y_train[4], all_x_test[4], all_y_test[4] = get_data_from_folder(".\\SpeechFolder\\Silence", types[4],
                                                                                        tts, max_data, (".wav"), extractor)
    x_train = np.empty((0, extractor.tot_features), float)
    y_train = np.empty((0, 1), str)
    x_test = np.empty((0, extractor.tot_features), float)
    y_test = np.empty((0, 1), str)
    for i in range(len(all_x_train)):
        x_train = np.append(x_train, all_x_train[i], axis=0)
        y_train = np.append(y_train, all_y_train[i], axis=0)
        x_test = np.append(x_test, all_x_test[i], axis=0)
        y_test = np.append(y_test, all_y_test[i], axis=0)

    x_train, x_test = standardize_data(x_train, x_test)
    print(x_train)
    print(y_test)
    np.savetxt("x_train.csv", x_train, delimiter=",")
    np.savetxt("y_train.csv", y_train, delimiter=",",fmt="%s")
    np.savetxt("x_test.csv", x_test, delimiter=",")
    np.savetxt("y_test.csv", y_test, delimiter=",",fmt="%s")

def train_classifiers():
    print("Reading in data")
    x_train = np.genfromtxt("x_train.csv", delimiter=",", dtype=float)
    y_train = np.genfromtxt("y_train.csv", delimiter=",", dtype=str)
    x_test = np.genfromtxt("x_test.csv", delimiter=",", dtype=float)
    y_test = np.genfromtxt("y_test.csv", delimiter=",", dtype=str)

    rf = RandomForestClassifier(n_estimators=100)
    svm_clf = SVC(C=7.5, gamma='scale')
    knn = KNeighborsClassifier(n_neighbors=10)
    print("Fitting classifiers")
    rf.fit(x_train, y_train.flatten())
    svm_clf.fit(x_train, y_train.flatten())
    knn.fit(x_train, y_train.flatten())

    print("Testing classifiers")
    test_classifier(rf, x_test, y_test)
    test_classifier(svm_clf, x_test, y_test)
    test_classifier(knn, x_test, y_test)

    #dump(rf, ".//" + type(rf).__name__ + "_classifier.joblib")
    #dump(svm_clf, ".//" + type(svm_clf).__name__ + "_classifier.joblib")
    #dump(knn, ".//" + type(knn).__name__ + "_classifier.joblib")

def real_world_test():
    clf = load("./SVC_classifier.joblib", )
    ss = load(".//standardizer.joblib")
    rwt = RealWorldTester(".//SpeechFolder//Brief_Test", clf, ss)
    rwt.plot_classification()

if __name__ == "__main__":

    get_new_data = True
    if get_new_data:
        extract_and_save()
    train_classifiers()




