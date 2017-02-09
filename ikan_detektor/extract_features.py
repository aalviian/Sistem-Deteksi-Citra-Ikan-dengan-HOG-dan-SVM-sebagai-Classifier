from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib

import glob
import os
from config import *

class EkstrakFitur(object):
    def __init__(self):
        super(EkstrakFitur, self).__init__()
        self.pathPositif = tesObjek
        self.pathNegatif = tesNotObjek
        # tipe deskriptor = HOG

    def startExtract(self):
        # Cek folder feature udah ada/belum, buat tempat untuk ekstraksi ciri
        if not os.path.isdir(fiturObjek):
            os.makedirs(fiturObjek)
        if not os.path.isdir(fiturBukanObjek):
            os.makedirs(fiturBukanObjek)

        print "Menghitung sample fitur positif (objek yang akan dideteksi)..."
        for im_path in glob.glob(os.path.join(self.pathPositif, "*")):
            im = imread(im_path, as_grey=True)
            # Hitung nilai HOG untuk mendapatkan fitur objek
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize) # variabel dari file config
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(fiturObjek, fd_name)
            joblib.dump(fd, fd_path)
        print "Positive features saved in {}".format(fiturObjek)

        print "Menghitung sample fitur negatif (citra yang bukan termasuk objek)..."
        for im_path in glob.glob(os.path.join(self.pathNegatif, "*")):
            im = imread(im_path, as_grey=True)
            # Hitung nilai HOG untuk mendapatkan fitur yang bukan objek
            fd = hog(im,  orientations, pixels_per_cell, cells_per_block, visualize, normalize)
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(fiturBukanObjek, fd_name)
            joblib.dump(fd, fd_path)
        print "Negative features saved in {}".format(fiturBukanObjek)

        print "Completed calculating features, yang akan digunakan untuk training oleh SVM"


