import csv
import random
import numpy

towrite  = open('noisyIris.csv', "wb")
filewriter = csv.writer(towrite, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

with open('iris.csv', 'rb') as csvfile:
	filereader = csv.reader(csvfile, delimiter=',')
	for row in filereader:
		features = row[:-1]
		for i in xrange(10):
			newRow = [float(i) for i in features] + numpy.random.uniform(0.1,0.2,4)
			newRow = numpy.append(newRow,row[-1])
			filewriter.writerow(newRow)

towrite.close()

