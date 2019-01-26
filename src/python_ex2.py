from Decision_tree_model import *
from KnnModel import KnnModel
from Naive_Bayes import *
import ut as ut


Classification_length = None


def main():
    trained, prop = read_from_train_file('train.txt')
    ut.init_features_dict(prop, trained)
    test = read_from_test_file('test.txt')

    model = KnnModel(5, Classification_length)
    list_after_calculate = model.run_model(trained, test)

    nb_model = Bayes(Classification_length)
    bayes_list = nb_model.run_model(trained, test)

    model = DecisionTreeModel(Classification_length, ut.Features_option)
    root = model.compute_DTL(trained, list(ut.Features_option.keys()), model.mode(trained), 0)
    tree = Tree(root)
    list_of_test_dist = model.run_model(test, tree)

    string_tree = tree.print_tree(root)
    with open("output_tree.txt", 'w') as file:
        file.write(string_tree)
    print_all(list_after_calculate, list_of_test_dist, bayes_list,test)


    """
    read_from_test_file function.
    read the test values and put them into the format
    of [ [(), (), ()....], [(), (), ()....] ]
    when each tuple is feature : value format
    """
def read_from_test_file(filename):
    file_ = open(filename, 'r+')
    list_of_values = []
    lines = file_.read().splitlines()
    first = True
    for line in lines:
        if first:
            feat_list = line.split("\t")
            first = False
        else:

            val = line.split("\t")
            row = []
            for i in range(len(val)):
                row.append((feat_list[i], val[i]))
            list_of_values.append(row)
    return list_of_values


"""
read_from_trained_file function.
read 
"""
def read_from_train_file(filename):
    global Classification_length
    list_of_values = []
    file_ = open(filename, 'r+')
    first = True
    lines = file_.read().splitlines()
    for line in lines:
        if first:
            temp = line.split("\t")
            feat_type = temp[:-1]
            Classification_length = len(temp) - 1
            first = False
        else:
            val = line.split("\t")
            row = []
            for i in range(len(val)):
                row.append((temp[i], val[i]))
            ut.init_class(val[i])
            list_of_values.append(row)
    return list_of_values, feat_type


"""
get_all_options_of_type function
get tbe all options of a given feature.
"""
def get_all_options_of_type(examples, index):
    options = set()
    for ex in examples:
        options.add(ex[index])
    return options

"""
print_all function.
write the all results of three models into an
output file.
"""
def print_all(knn_list, dt_list,bayes, test):
    file = open('output.txt', 'w')
    counter = 1
    lines = []
    lines.append("Num\tDT\tKNN\tnaiveBayes")
    for k, de, ba in zip(knn_list, dt_list, bayes):
        lines.append(str(counter) + "\t" + de[3][ut.FEATURE_VALUE] + "\t" + k[3][ut.FEATURE_VALUE] +"\t" + ba[3][ut.FEATURE_VALUE])
        counter += 1
    knn_acc = str(get_accuracy(knn_list, test))
    dt_acc = str(get_accuracy(dt_list, test))
    nb_acc = str(get_accuracy(bayes, test))
    lines.append("\t" + knn_acc + "\t" + dt_acc + "\t" + nb_acc)
    file.writelines("\n".join(lines))
    file.close()

"""
get_accuracy function.
compute the accuracy of the given list to the original one.
"""
def get_accuracy(list_1, origin_list):
    number_of_good_values = 0
    for att1, att_origin in zip(list_1, origin_list):
        if att1[Classification_length][1] == att_origin[Classification_length][1]:
            number_of_good_values += 1
    temp = number_of_good_values / len(origin_list)
    return math.ceil(temp*100) / 100


if __name__ == '__main__':
    main()