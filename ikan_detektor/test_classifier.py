# -*- coding: utf-8 -*-
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
from nms import nms
from config import *
import numpy as np

class DeteksiObjek(object):
    def __init__(self, objekCitra, downscale=2.5, visualize_det=True):
        super(DeteksiObjek, self).__init__()
        self.objekCitra = objekCitra
        self.downscale = downscale
        self.visualize_det = visualize_det
        self.minWindowSize = minWindowSize
        self.stepSize = stepSize

    def sliding_window(self, image, window_size, stepSize):
        for y in xrange(0, image.shape[0], stepSize[1]):
            for x in xrange(0, image.shape[1], stepSize[0]):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

    def startDeteksiObjek(self):
        im = imread(self.objekCitra, as_grey=False)
        # Load model SVM
        clf = joblib.load(folderModel)

        detections = []
        scale = 0
        for im_scaled in pyramid_gaussian(im, downscale=self.downscale):
            cd = []
            if im_scaled.shape[0] < self.minWindowSize[1] or im_scaled.shape[1] < self.minWindowSize[0]:
                break
            # print self.sliding_window(im_scaled, self.minWindowSize, self.stepSize)

            # start sliding windows, untuk scan bagian image sesuai ukuran rectangle
            for (x, y, im_window) in self.sliding_window(im_scaled, self.minWindowSize, self.stepSize):
                # print x,y,im_window
                if im_window.shape[0] != self.minWindowSize[1] or im_window.shape[1] != self.minWindowSize[0]:
                    continue
                # Hitung hog dari bagian sliding windows
                fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
                fd = np.array(fd).reshape((1, -1))
                prediksi = clf.predict(fd)
                if prediksi == 1:
                    # Tandai objek yang terdeteksi
                    print  "Objek terdeteksi:: Lokasi -> ({}, {})".format(x, y)
                    print "Scale ->  {} | Confidence Score {} \n".format(scale,clf.decision_function(fd))
                    detections.append((x, y, clf.decision_function(fd), int(self.minWindowSize[0]*(self.downscale**scale)), int(self.minWindowSize[1]*(self.downscale**scale))))
                    cd.append(detections[-1])

                if self.visualize_det:
                    clone = im_scaled.copy()
                    for x1, y1, _, _, _  in cd:
                        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 + im_window.shape[0]), (0, 0, 0), thickness=2)
                    cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y + im_window.shape[0]), (255, 255, 255), thickness=2)
                    cv2.imshow("Proses scan untuk mendeteksi objek", clone)
                    cv2.waitKey(10)
                    # print "masuk massss"
            # Perkecil image, agar objek yg besar bisa terdeteksi
            scale+=1

        clone = im.copy()
        for (x_tl, y_tl, _, w, h) in detections:
            cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 0, 0), thickness=2)
        cv2.imshow("Deteksi mentah, rectangle masih menumpuk", im)
        cv2.waitKey()

        # Perform Non Maxima Suppression, buat ngilangin rectangle yg menumpuk
        detections = nms(detections, threshold)

        jumlahObjek = 0
        for (x_tl, y_tl, _, w, h) in detections:
            cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (255, 0, 0), thickness=2)
            jumlahObjek=jumlahObjek+1
        print "Jumlah objek yang terdeteksi =",jumlahObjek
        cv2.imshow("Deteksi Akhir, menghilangkan rectangle yang saling menumpuk", clone)
        cv2.waitKey()
        

