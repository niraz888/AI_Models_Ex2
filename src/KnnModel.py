import ut as ut
"""
KnnModel class.
model that tries to classify attributes according to 
examples, and more specificly - the k (in our case 5) closest
examples
"""
class KnnModel(object):

    def __init__(self, k, ind):
        self.k_number = k
        self.class_index = ind
    """
    calculate_hamming_distance method.
    :parameter att1: attribute
    :parameter att2: att
    """

    def calculate_hamming_distance(self, att1, att2):
        dist = 0
        for i in range(self.class_index):
            if att1[i][1] != att2[i][1]:
                dist += 1
        return dist
    """
    predict method.
    predict the classification of a single attribute
    :param predict_att: the att we want to classify
    :param trained: all examples
    :return the predicted classification
    """
    def predict(self, predict_att, trained):

        ham_tuple_list = []
        if len(trained) < self.k_number:
            return
        for example in trained:
            num = self.calculate_hamming_distance(predict_att, example)
            ham_tuple_list.append((num, example[self.class_index][1]))
        # sort the list by the first value in tuple -> hamming dist
        sorted_list = sorted(ham_tuple_list, key=lambda x: x[0])
        yes = 0
        no = 0
        for i in range(self.k_number):
            if sorted_list[i][ut.FEATURE_VALUE] == ut.ret_spec_class('yes'):
                yes += 1
            elif sorted_list[i][ut.FEATURE_VALUE] == ut.ret_spec_class('no'):
                no += 1
        if yes >= no:
            return 'yes'
        else:
            return 'no'
    """
    run_model method.
    classify all the test attributes by the KNN model
    and according to the trained list
    :param trained_list: the trained_list examples
    :param test_list: the the att we want to classify
    """
    def run_model(self, trained_list, test_list):
        new_list = []
        for e in test_list:
            item = []
            for i in range(self.class_index):
                item.append((e[i][ut.FEATURE_TYPE], e[i][ut.FEATURE_VALUE]))
            item.append((e[i], self.predict(e, trained_list)))
            new_list.append(item)
        return new_list












