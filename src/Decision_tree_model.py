import math
import ut as ut

"""
Tree class.
represent a DT that has a root, and is run by the model.
"""


class Tree(object):
    def __init__(self, root):
        self.root = root

    def resolve_node(self, child_map, att):
        for ch in child_map:
            if ch[0] == att:
                return ch[1]
    """
    print_tree method.
    build a string that represent the DT structure.
    """
    def print_tree(self, node):
        tree_string_representation = ""
        sorted_map = sorted(node.child_map)
        for child in sorted_map:
            tree_string_representation += node.depth * "\t"
            if node.depth > 0:
                tree_string_representation += "|"
            tree_string_representation += node.attribute + "=" + child[0]
            curr = child[1]
            if curr.is_leaf:
                tree_string_representation += ":" + curr.default + "\n"
            else:
                tree_string_representation += "\n" + self.print_tree(curr)
        return tree_string_representation
    """
    resolve_index method.
    :returns the index of a given feature type.
    """
    def resolve_index(self, name, ex):
        i = 0
        for i in range(len(ex)):
            if name == ex[i][ut.FEATURE_TYPE]:
                return i
            i += 1
        return -1
    """
    resolve_tree method.
    by a given root and a single example, return the
    node that represent the appropriate classification of example
    """
    def resolve_tree(self, single_example, root):
        curr = Node(root.depth, root.is_leaf, root.child_map, root.attribute, root.default)
        if curr.is_leaf:
            return curr.default
        for ex in curr.child_map:
            val = ex[1]
            i = self.resolve_index(curr.attribute, single_example)
            if ex[0] == single_example[i][1]:
                curr = val
                break
        return self.resolve_tree(single_example, curr)


"""
Node class.
represent a node ina DT.
depth - depth of node.
is_leaf - boolean value says if it is leaf
child_map - a list that mapps all node's children
attribute - attirbute of the node
default - default classification of node
"""


class Node(object):
    def __init__(self, depth, is_leaf_, map_, att, default):
        self.depth = depth
        self.is_leaf = is_leaf_
        self.child_map = map_
        self.attribute = att
        self.default = default

        """
        DecisionTreeModel class.
        build a decision tree according the the know DTL algorithm
        """


class DecisionTreeModel(object):
    def __init__(self, num, dict):
        self.class_index = num
        self.feat_dict = dict
    """
    comute_DTL method.
    compute the well known DTL algorithm that return the root of the
    decision tree.
    :param examples: list of all examples
    :param list_of_attributes: list of possible attributes
    :param default: default calssification
    :param depth: depth of node
    :return root of the tree
    """
    def compute_DTL(self, examples, list_of_attributes, default, depth):
        if len(examples) == 0:
            return Node(depth, True, None, None, default)
        elif self.check_if_all_classification_are_same(examples):
            return Node(depth, True, None, None, examples[0][self.class_index][ut.FEATURE_VALUE])
        elif len(list_of_attributes) == 0:
            return Node(depth, True, None, None, self.mode(examples))
        else:
            child_list = []
            best_att = self.choose_best_attribute(list_of_attributes, examples)
            child_att = list_of_attributes[:]
            child_att.remove(best_att)
            curr_root = Node(depth, False, child_list, best_att, None)
            all_posibilites_of_given_feature = sorted(ut.Features_option[best_att])
            for feat_value in all_posibilites_of_given_feature:
                sub_exm = self.get_distribution_of_feature_val(examples, feat_value, best_att)
                child_list.append((feat_value, self.compute_DTL(sub_exm, child_att, default, depth+1)))
        return curr_root
    """
    choose_best_attribute method.
    choose the best attirbute -> the attirbute with
    the clearest distirbution.
    :param list_of_att: list of the current relevant features
    :param examples: list of current relevans examples
    :return best att option
    """
    def choose_best_attribute(self, list_of_att, examples):
        best_curr_feat = None
        max_g = -100
        entropy_of = self.entropy(examples)
        for feat in list_of_att:
            gain = self.gain_information(entropy_of, examples, feat)
            if gain > max_g:
                max_g = gain
                best_curr_feat = feat
        return best_curr_feat
    """
    mode method.
    return the majority between two classification
    :param examples: list of current examples
    :return bool
    """
    def mode(self, examples):
        true_ = 0
        false_ = 0
        for ex in examples:
            ans = ex[self.class_index][ut.FEATURE_VALUE]
            if ans == ut.ret_spec_class('yes'):
                true_ += 1
            else:
                false_ += 1
        if true_ >= false_:
            return ut.ret_spec_class('yes')
        else:
            return ut.ret_spec_class('no')
    """
    entropy method.
    calculate entropy.
    """
    def entropy(self, list_of_examples):
        yes = self.get_propability(list_of_examples, ut.ret_spec_class('yes'))
        no = self.get_propability(list_of_examples, ut.ret_spec_class('no'))
        if yes == 0 or no == 0:
            return 0
        res = -yes * math.log(yes, 2) - no * math.log(no, 2)
        return res
    """
    gain_information method.
    """
    def gain_information(self, entropy, examples, feature):
        sum = 0
        temp = sorted(list(self.feat_dict[feature]))
        for feat_val in temp:
            sub_example = self.get_distribution_of_feature_val(examples, feat_val, feature)
            percent = float(len(sub_example)) / float(len(examples))
            ent = self.entropy(sub_example)
            sum += (percent * ent)
        return entropy - sum
    """
    check_if_all_classification_are_same method.
    """
    def check_if_all_classification_are_same(self, examples):
        tag = examples[0][self.class_index][ut.FEATURE_VALUE]
        for ex in examples:
            if ex[self.class_index][ut.FEATURE_VALUE] != tag:
                return False
        return True
    """
    get_distirbution_of_feature_val method.
    get the distirbution of the classification of a
    specific feature value.
    """
    def get_distribution_of_feature_val(self, examples, feat_val, feature):
        sub = []
        # resolve the index of the desired feature
        index = self.resolve_index_by_feature(feature, examples[0])
        for ex in examples:
            if ex[index][ut.FEATURE_VALUE] == feat_val:
                sub.append(ex)
        return sub
    """
    resolve_index_by_feature method.
    :returns the index that represent the specific feature.
    """
    def resolve_index_by_feature(self, feature, single_example):
        i = 0
        for ex in single_example:
            if ex[ut.FEATURE_TYPE] == feature:
                return i
            i += 1
    """
    get_propability method.
    get the propability of a given classification
    """
    def get_propability(self, list_of_examples, classification):
        count = 0
        if len(list_of_examples) == 0:
            return 0
        for ex in list_of_examples:
            if ex[self.class_index][ut.FEATURE_VALUE] == classification:
                count += 1
        return count / len(list_of_examples)
    """
    run_model method.
    run on a test list of examples the DT model based on the
    tree that was built.
    """
    def run_model(self,test_list, tree):
        new_list = []
        for e in test_list:
            item = []
            for i in range(self.class_index):
                item.append((e[i][ut.FEATURE_TYPE], e[i][ut.FEATURE_VALUE]))
            item.append((e[i+1][ut.FEATURE_TYPE], tree.resolve_tree(item, tree.root)))
            new_list.append(item)
        return new_list