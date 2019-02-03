try:
	import pyimageocr
except Exception as e:
	print("Error...", e)

print('Character : Accuracy')

ocr = pyimageocr.OCR(mode='en')
# ocr.train("Training")
print("Trainning Done !!")
a = ocr.pattern_match(file="Test/1-img015-00045.png")
print(a)
# ocr.tlableshresoldImage()
# ocr.getimageHistogram()
# ocr.imageShow()
# getImageFormFile
