# Print function may not work if proper GUI is not selected
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
print("Using OpenCV backend")
class OCR:
    """
    OCR  class
    """
    __mode = ""
    __codename = "pyimageocr"
    __ver = 1.0
    __image = []
    __height, __width, __channels = 0, 0, 0
    rowSegment = []
    colSegment = []

    def __init__(self, mode = "en"):
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.cache_file = "__pycache__"
        if not os.path.exists(self.cache_file):
            os.makedirs(self.cache_file)
        self.__mode = mode

    def train(self, folder, save="train.bin"):
        sub_folder = os.listdir(folder)
        training_images = [] 
        training_label = []
        i = 0

        knn = cv2.ml.KNearest_create()
        for class_lable in sub_folder:
            i += 1
            cu_folder = folder + os.sep + class_lable
            imgs = os.listdir(cu_folder)
            tmp = []
            print(cu_folder)
            for img in imgs:
                char_image = cv2.imread(cu_folder+os.sep+img)
                char_image = 255 - cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
                char_image = cv2.resize(char_image, (64, 64)).astype(np.float32)
                char_image = char_image.flatten()
                training_images.append(char_image)
                training_label.append([i])

        training_images = np.array(training_images, dtype=np.float32)
        training_label = np.array(training_label, dtype=np.float32)

        # print("training_images : ", training_images.shape)
        # print("training_label", training_label.shape)

        train = knn.train(training_images, cv2.ml.ROW_SAMPLE, training_label)
        np.savez(save,train=training_images, train_labels=training_label)



    def pridict(self, filename):
    	self.getImageFormFile(filename)
    	self.thresoldImage()
    	return self.__Segment()

    def getImageFormFile(self, filename):
        try:
            img = cv2.imread(filename)
            self.__image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.__height, self.__width, self.__channels = img.shape
        except Exception:
            print("File Read Error... (Line 24)")

    def thresoldImage(self):
        try:
            sum = 0
            for i in range(0, self.__height):
                for j in range(0, self.__width):
                    sum = sum+ self.__image[i, j]
            thresh = sum/(self.__width*self.__height)
            self.__image = cv2.threshold(self.__image, thresh, 255, cv2.THRESH_BINARY)[1]
        except Exception:
            print("Unknown Execption at line 34")

    def imageShow(self):
        try:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', self.__image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            print("System can't detect any compatible version of OpenCV")

    def compressInt(self, number):
        remove  = []
        for i in range(0, len(number)-1):
            if abs(number[i] - number[i+1]) < 3:
                remove.append(number[i+1])
        for i in range(0, len(remove)):
            number.remove(remove[i])
                
        return number
    def getNoOfFile(self, path):
        path, dirs, files = os.walk(path).__next__()
        return len(files)
    
    def save(self, image, path):
        num = str(self.getNoOfFile(path)+1)
        cv2.imwrite(self.cache_file+"/tmp.jpg",image)
        main = cv2.imread(self.cache_file+"/tmp.jpg")
        main = cv2.resize(main, (64, 64))
        cv2.imwrite(path+"/"+num+".jpg",main)

    def pattern_match(self, image=None, file=None):
        if file:
            main = cv2.imread(file)
        else:
            cv2.imwrite(self.cache_file+"/tmp.jpg",image)
            main = cv2.imread(self.cache_file+"/tmp.jpg")
        main = cv2.resize(main, (64, 64))
        main = cv2.cvtColor(main, cv2.COLOR_BGR2GRAY)
        main = np.array([main.flatten()], np.float32)

        #Load the kNN Model
        with np.load('train.bin.npz') as data:
            train = data['train']
            train_labels = data['train_labels']

        knn = cv2.ml.KNearest_create()
        knn.train(train, cv2.ml.ROW_SAMPLE, train_labels) 
        ret, result, neighbours, dist = knn.findNearest(main,k=1)
        return self.classes[int(result)-1]

    def __getRows(self, bit_factor=5):
        strip = []
        start = False
        tmp = 0
        loop = 0
        shaped = cv2.resize(self.__image, (bit_factor, self.__height))
        for i in shaped:
            loop += 1
            if sum(i) < bit_factor*255:
                if not start:
                    start = True
                    tmp = loop
            if sum(i) == bit_factor*255:
                if start:
                    start = False
                    strip.append((tmp,loop))
        return strip

        
    def __getWord(self, image, bit_factor = 10):
        height, width = image.shape
        strip = []
        start = False
        tmp = 0
        loop = 0

        shaped = shaped = cv2.resize(image, (self.__width, bit_factor))
        for i in zip(*shaped):
            loop += 1
            if sum(i) < bit_factor*255:
                if not start:
                    start = True
                    tmp = loop
            if sum(i) == bit_factor*255:
                if start:
                    start = False
                    strip.append((tmp, loop))
        buff = ""
        for i, j in strip:
            buff = buff + self.pattern_match(image=image[0:height, i:j])
        
        return buff
    def __Segment(self):
        line = []
        self.rowSegment = self.__getRows()
        for i in range(len(self.rowSegment)):
            line.append(self.__getWord(self.__image[self.rowSegment[i][0]:self.rowSegment[i][1], 0:self.__width]))
        return line

def main():
    ocr = OCR(mode='en')
    ocr.train("../Training")


if __name__ == '__main__':
    main()
