import utils
import math as m


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


class NaiveBayes(classification_algorithm):
    '''
    naive bayes classifer
    '''

    def __init__(self):
        classification_algorithm.__init__(self)
        self.classes_amount = 0
        self.smoothing_k = None

    def fit(self, X, y):
        '''
        fitting the data naive bayes model
        :param X: data
        :param y: labels
        '''
        self.X = list(X)
        self.y = list(y)
        self.attributes_len = len(X[0])
        self.X_size = len(self.X)
        self.classes, self.classes_labels_counter = utils.extract_unique_labels_and_labels_counter(self.y)
        self.classes_amount = len(self.classes)
        self.smoothing_k = [0] * self.attributes_len
        for i in range(self.attributes_len):
            sub_attribute, temp = utils.extract_unique_labels_and_labels_counter([x[i] for x in self.X])
            self.smoothing_k[i] = len(sub_attribute)

    def _predict_single_sample(self, sample):
        '''
        :param sample: predicting label for sample
        :return: the classification
        '''
        attribute_instances_counter_by_label = [[0] * self.classes_amount for i in range(self.attributes_len)]
        # predicting all the samples
        for i, X_sample in enumerate(self.X):
            for j, cls in enumerate(self.classes):
                if self.y[i] == cls:
                    for k in range(self.attributes_len):
                        if sample[k] == X_sample[k]:
                            attribute_instances_counter_by_label[k][j] += 1
                    break

        for i in range(self.attributes_len):
            for j in range(self.classes_amount):
                attribute_instances_counter_by_label[i][j] = (attribute_instances_counter_by_label[
                                                                  i][j] + 1) / (
                                                                 self.classes_labels_counter[j] + self.smoothing_k[i])

        classes_prior = [x / self.X_size for x in self.classes_labels_counter]

        for i in range(len(self.classes)):
            for j in range(self.attributes_len):
                classes_prior[i] *= attribute_instances_counter_by_label[j][i]

        return self.classes[utils.argmax(classes_prior)]
