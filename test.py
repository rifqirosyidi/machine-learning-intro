import pandas
import numpy
import sklearn

data = pandas.read_csv("student-mat.csv", sep=";")
data = data[['G1', 'G2', 'G3', 'studytime', 'absences', 'failures']]

data_to_predict = 'G3'

X = numpy.array(data.drop([data_to_predict], 1))
Y = numpy.array(data[data_to_predict])

x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
