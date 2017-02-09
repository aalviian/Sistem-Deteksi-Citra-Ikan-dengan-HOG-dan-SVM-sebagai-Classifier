#!/usr/bin/env python
from ikan_detektor.extract_features import EkstrakFitur
from ikan_detektor.train_classifier import TrainingData
from ikan_detektor.test_classifier import DeteksiObjek

# Masih pake data citra mobil, belum dapet data buat ikan

isTrain = raw_input("Sudah ada model SVM? (y/n): ")
if isTrain=="n":
	# Mengekstraksi ciri, untuk mendeskripsikan apa yang akan dideteksi,
	# dalam hal ini memisahkan objek dengan background. (Proses dekripsi menggunakan HOG)
	print "Step 1: Ekstrak fitur dari data latih citra, untuk memisahkan klasifikasi antara objek yang dideteksi dengan backgroundnya"
	EkstrakFitur().startExtract()

	# Proses pelatihan data untuk mendapatkan model SVM
	print "\nStep 2: Proses training untuk klasifikasi SVM"
	TrainingData().startTraining()

# tes objek citra yg akan dideteksi ikannya
print "\nStep 3: Proses mendeteksi objek dari model yang telah dilatih"
nomor = raw_input("Pilih nomor urut citra: ")
tesCitra = "data/datalatih/CarData/TestImages/test-"+str(nomor)+".pgm"
# tesCitra = "data/datalatih/Ikan/dataTES.jpg"
deteksi = DeteksiObjek(tesCitra)
deteksi.startDeteksiObjek()
