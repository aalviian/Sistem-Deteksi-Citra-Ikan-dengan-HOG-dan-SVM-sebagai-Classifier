'''
Set variabel utama yang digunakan pada kelas
'''

# Variabel yang digunakan dalam proses HOG
minWindowSize = [100, 40]
stepSize = [10, 10]
orientations = 9
pixels_per_cell = [8, 8]
cells_per_block = [3, 3]
visualize = False
normalize = True

# Folder data latih
tesObjek = "data/datalatih/CarData/TrainImages/pos" 	# tesPositif->folder yg berisi data latih berupa objek yang akan dideteksi
tesNotObjek = "data/datalatih/CarData/TrainImages/neg"	# tesNegatif->folder yg berisi data latih berupa background dari objek

# Folder tempat vektor hasil ekstrak ciri dan model
fiturObjek = "data/fitur/pos"
fiturBukanObjek = "data/fitur/neg"
folderModel = "data/models/svm.model"

# nilai thresold untuk proses NMS
threshold = 0.3
