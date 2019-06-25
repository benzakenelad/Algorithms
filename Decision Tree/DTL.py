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


class DecisionTree(classification_algorithm):
    '''
    Decision Tree classifier
    '''

    def __init__(self):
        classification_algorithm.__init__(self)
        self.attributes = None
        self.prior = None
        self.tree_root = None
        self.attributes_map = None
        self.attributes_to_subattributes_map = None

    def fit(self, X, y, attributes):
        '''
        fitting the data Decision Tree model
        :param X: data
        :param y: labels
        :param attributes: attributes list (for tree printing)
        '''
        self.X = list(X)
        self.y = list(y)
        self.X_size = len(self.X)
        self.attributes = attributes
        self.attributes_len = len(self.X[0])
        self.attributes_to_subattributes_map = dict()
        for i in range(self.attributes_len):
            temp_values, temp_values_counter = utils.extract_unique_labels_and_labels_counter([x[i] for x in self.X])
            self.attributes_to_subattributes_map.update({self.attributes[i]: temp_values})
        self.classes, self.classes_labels_counter = utils.extract_unique_labels_and_labels_counter(self.y)
        self.prior = self.classes[utils.argmax(self.classes_labels_counter)]
        self.tree_root = self.__DTL(self.X, self.y, self.attributes, self.prior)
        self.attributes_map = {att: i for i, att in enumerate(attributes)}

    def _predict_single_sample(self, sample):
        '''
        :param sample: predicting label for sample
        :return: the classification
        '''
        tree = self.tree_root
        while True:
            if tree.classification is not '':
                return tree.classification
            current_attribute_index = self.attributes_map.get(tree.subtrees[0].attribute)
            for sub_tree in tree.subtrees:
                if sub_tree.value == sample[current_attribute_index]:
                    tree = sub_tree
                    break
            else:
                return self.prior

    def __information_gain(self, attribute_data, labels):
        '''
        calculating the information gain using entropy
        :param attribute_data:
        :param labels:
        :return:
        '''
        if len(attribute_data) == 0 or len(attribute_data) != len(labels):
            raise Exception('illegal x,y size')
        n = len(attribute_data)
        # counting sub attribute instances and extracting the values
        sub_attribute_values, attributes_counter = utils.extract_unique_labels_and_labels_counter(attribute_data)
        labels_values, labels_counter = utils.extract_unique_labels_and_labels_counter(labels)
        sub_attributes_classes_counter = [[0] * len(labels_values) for i in range(len(sub_attribute_values))]
        sub_attribute_map = {k: v for v, k in enumerate(sub_attribute_values)}
        classes_map = {k: v for v, k in enumerate(labels_values)}

        # counting attributes instances for entropy calculation
        for i, x in enumerate(attribute_data):
            sub_attributes_classes_counter[sub_attribute_map.get(x)][classes_map.get(labels[i])] += 1

        # calculating entropy
        sub_attributes_entropy = [0] * len(sub_attribute_values)
        conditioned_entropy = 0
        for i in range(len(sub_attribute_values)):
            sub_attributes_entropy[i] = self.__entropy(sub_attributes_classes_counter[i])
            conditioned_entropy += (sum(sub_attributes_classes_counter[i]) / n) * sub_attributes_entropy[i]

        labels_entropy = self.__entropy(labels_counter)
        return labels_entropy - conditioned_entropy

    def __entropy(self, x):
        '''
        calculating vector x entropy
        :param x:
        :return:
        '''
        n = sum(x)
        return -sum([(a / n) * m.log(a / n, 2) for a in x if a != 0])

    def __DTL(self, data, labels, attributes, default):
        '''
        DTL method recursively building the Decision Tree nodes
        :param data:
        :param labels: labels list
        :param attributes: attributes list
        :param default: default classification
        :return:
        '''
        tree = Tree()

        # stoping criteria
        if len(data) == 0:
            # no more samples
            tree.classification = default
            return tree
        if len(set(labels)) == 1:
            # all the examples from the same class
            tree.classification = labels[0]
            return tree
        if len(attributes) == 0:
            # no more attributes
            label_values, label_values_counter = utils.extract_unique_labels_and_labels_counter(labels)
            tree.classification = label_values[utils.argmax(label_values_counter)]
            return tree

        # choosing an attribute with the highest information gain
        best_att_value, best_att_index = self.__choose_attribute(data, labels, attributes)
        best_att_sub_values = self.attributes_to_subattributes_map.get(best_att_value)

        # calculating default values
        classes, classes_labels_counter = utils.extract_unique_labels_and_labels_counter(labels)
        default = classes[utils.argmax(classes_labels_counter)]

        # recursively building the next branches

        for i, sub_att in enumerate(best_att_sub_values):
            new_data = []
            new_labels = []

            for j, sample in enumerate(data):
                if sample[best_att_index] == sub_att:
                    new_data.append(data[j])
                    new_labels.append(labels[j])

            # removing the attribute's column data
            if len(new_data) != 0:
                new_data = utils.remove_column(new_data, best_att_index)
            # removing the attribute
            new_attributes = list(attributes)
            new_attributes = [new_attributes[i] for i in range(len(new_attributes)) if i != best_att_index]
            # building the next node
            sub_tree = self.__DTL(new_data, new_labels, new_attributes, default)
            # setting the new node information
            sub_tree.value = sub_att
            sub_tree.attribute = best_att_value

            tree.subtrees.append(sub_tree)
        return tree

    def __choose_attribute(self, data, labels, attributes):
        '''
        select the attribute with the maximum information gain
        :param data:
        :param labels:
        :param attributes:
        :return: the attribute value and the attribute index where the information gain is maximized
        '''
        info_gain = [0] * len(attributes)
        for i, att in enumerate(attributes):
            info_gain[i] = self.__information_gain([x[i] for x in data], labels)
        return attributes[utils.argmax(info_gain)], utils.argmax(info_gain)

    def print_tree(self, output=None):
        '''
        print the tree to output file
        :param output:
        :return:
        '''
        if self.tree_root is None:
            raise Exception('tree is not exist, please run fit method')

        self.tree_root.subtrees.sort(key=lambda x: x.value)
        for sub_tree in self.tree_root.subtrees:
            self.__sub_tree_print(sub_tree, 0, output)

    def __sub_tree_print(self, tree, tabs_amount, output):
        '''
        print the subtree to output file
        :param tree:
        :param tabs_amount:
        :param output:
        :return:
        '''
        line = ''
        for i in range(tabs_amount): line += '\t'
        classifications = []
        if tree.classification is '':
            if tabs_amount > 0:
                line = '{}|{}={}\n'.format(line, tree.attribute, tree.value)
            else:
                line = '{}{}={}\n'.format(line, tree.attribute, tree.value)
        else:
            if tabs_amount > 0:
                line = '{}|{}={}:{}\n'.format(line, tree.attribute, tree.value, tree.classification)
            else:
                line = '{}{}={}:{}\n'.format(line, tree.attribute, tree.value, tree.classification)
        if output is None:
            print(line)
        else:
            output.write(line)
        tree.subtrees.sort(key=lambda x: x.value)
        for sub_tree in tree.subtrees:
            self.__sub_tree_print(sub_tree, tabs_amount + 1, output)


class Tree:
    '''
    tree node class
    '''

    def __init__(self):
        self.subtrees = []
        self.value = ''
        self.attribute = ''
        self.classification = ''
