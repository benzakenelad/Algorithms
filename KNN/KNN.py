import math as m
import utils


class classification_algorithm:
    '''
    abstract class that unified all the shared data
    '''

    def __init__(self):
        self.X = None
        self.y = None
        self.classes = None
        self.classes_label_counter = None
        self.attributes_len = 0
        self.X_size = 0

    def predict(self, X_test):
        '''
        predicting set of samples
        :param X_test: samples
        :return: classifications list
        '''
        predictions = []
        for x in X_test:
            predictions.append(self._predict_single_sample(x))
        return predictions

    def _predict_single_sample(self, sample):
        '''
        abstract method
        :param sample: predicting label for sample
        :return: the classification
        '''
        raise NotImplementedError()


class KNN(classification_algorithm):
    '''
    K-nearest neighbor classifier
    '''

    def __init__(self, k=5):
        classification_algorithm.__init__(self)
        self.k = k
        self.classes_map = None

    def fit(self, X, y):
        '''
        fit a KNN model to the data
        :param X: data
        :param y: label
        '''
        self.X = list(X)
        self.y = list(y)
        self.classes, self.classes_labels_counter = utils.extract_unique_labels_and_labels_counter(self.y)
        self.attributes_len = len(X[0])
        self.classes_map = {k: v for v, k in enumerate(self.classes)}

    def _predict_single_sample(self, sample):
        '''
        :param sample: predicting label for sample
        :return: the classification
        '''
        if self.X is None:
            raise Exception('KNN data was not initialize with fit method.')
        if self.attributes_len != len(sample):
            raise ValueError('sample len is not as DATA\'s len')

        hamming_distances = []
        for x in self.X:
            hamming_distances.append(self.__distance(sample, x))
        # argsort is the indices of the sorted hamming distances
        argsort = utils.argsort(hamming_distances)[:self.k]
        predictions = [self.y[i] for i in argsort]
        predictions_class_counter = [0] * len(self.classes)

        for p in predictions:
            predictions_class_counter[self.classes_map.get(p)] += 1

        return self.classes[utils.argmax(predictions_class_counter)]

    def __distance(self, sample_a, sample_b):
        '''
        calculating hamming distance between sample_a and sample_b
        :param sample_a:
        :param sample_b:
        :return: the distance
        '''
        hamming_distance = 0
        for i in range(self.attributes_len):
            if sample_a[i] != sample_b[i]:
                hamming_distance += 1
        return hamming_distance
