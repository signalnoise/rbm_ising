import os

path = "../rename folder/trained_rbm.pytorch."

for i in range (1010, -10, -10):
	os.rename(path + str(i-4000), path + str(i+4000).zfill(5))
