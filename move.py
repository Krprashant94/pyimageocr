import os

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
char = -1
k = 0
source = "English Font Image"
dist = "Training"
for sample, _, files in os.walk(source):
	i = 0
	if not os.path.exists(dist+os.sep+classes[char%26]):
		os.mkdir(dist+os.sep+classes[char%26])
	if char%26 == 0:
		k +=1 
	for file in files:
		
		
		if i % 4 == 0:
			os.rename(os.path.join(sample, file), dist+os.sep+classes[char%26]+os.sep+str(k)+'-'+file)
		i += 1
	char += 1
