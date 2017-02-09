# -*- coding: utf-8 -*-
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import argparse as ap
import glob
import os
from config import *

class TrainingData(object):
    def __init__(self):
        super(TrainingData, self).__init__()
        self.klasifikasiPositif = fiturObjek
        self.klasifikasiNegatif = fiturBukanObjek
        # tipe klasifier, linear SVM

    def startTraining(self):
        dataVektor = []
        kelas = []
        # Load fitur objek
        for fiturVektor in glob.glob(os.path.join(self.klasifikasiPositif,"*.feat")):
            fd = joblib.load(fiturVektor)
            dataVektor.append(fd)
            kelas.append(1) # kelas 1 menandakan objek

        # Load fitur bukan objek
        for fiturVektor in glob.glob(os.path.join(self.klasifikasiNegatif,"*.feat")):
            fd = joblib.load(fiturVektor)
            dataVektor.append(fd)
            kelas.append(0) # kelas 0 menandakan bukan objek

        # Mulai training data dari fitur positif dan negatif HOG sebelumnya
        clf = LinearSVC()
        print "Training a Linear SVM Classifier..."
        clf.fit(dataVektor, kelas)

        if not os.path.isdir(os.path.split(folderModel)[0]):
            os.makedirs(os.path.split(folderModel)[0])

        print "Training sukses."
        print "Spesifikasi model SVM yang telah dilatih: "
        print clf
        joblib.dump(clf, folderModel)
        print "Model classifier saved to {}".format(folderModel)

