import ut as ut
"""
Bayes class.
represent model that classify based on the bayes formula
"""
class Bayes(object):
    def __init__(self, ind):
        self.class_index = ind
    """
    create_tags_list method.
    create two list of examples that classify as yes or no, 
    and map them into a dict
    """
    def create_tags_list(self, examples):
        global Yes_Prob, No_Prob
        dist_dict = {}
        y_list = []
        n_list = []
        for ex in examples:
            item = []
            for i in range(self.class_index):
                item.append((ex[i][ut.FEATURE_TYPE], ex[i][ut.FEATURE_VALUE]))
            if ex[i+1][ut.FEATURE_VALUE] == ut.ret_spec_class('yes'):
                y_list.append(item)
            elif ex[i+1][ut.FEATURE_VALUE] == ut.ret_spec_class('no'):
                n_list.append(item)
        dist_dict["yes"] = y_list
        dist_dict["no"] = n_list
        length = len(y_list) + len(n_list)
        Yes_Prob = float(len(y_list) / length)
        No_Prob = 1 - Yes_Prob
        return dist_dict
    """
    resolve_index method.
    resolve the index of a given name.
    """
    def resolve_index(self, name, ex):
        i = 0
        for i in range(len(ex)):
            if name == ex[i][ut.FEATURE_TYPE]:
                return i
            i += 1
        return -1
    """
    run_model mehtod.
    classify every test example based on the bayes model
    """
    def run_model(self, trained_list, test_list):
        distribution_list = self.create_tags_list(trained_list)
        new_list = []
        for e in test_list:
            item = []
            for i in range(self.class_index):
                item.append((e[i][ut.FEATURE_TYPE], e[i][ut.FEATURE_VALUE]))
            item.append((e[i+1][ut.FEATURE_TYPE], self.predict_single_example(e, distribution_list)))
            new_list.append(item)
        return new_list


    """
    predict_single_example method.
    predict classification of a single example, by the bayes formula.
    """
    def predict_single_example(self, example, tags):
        yes_result_dict = {}
        no_result_dict = {}
        for i in range(self.class_index):
            count_yes = 0
            pos_att = ut.Features_option[example[i][ut.FEATURE_TYPE]]
            for t in tags["yes"]:
                if example[i][ut.FEATURE_VALUE] == t[i][ut.FEATURE_VALUE]:
                    count_yes += 1
            curr_feat_val = example[i][ut.FEATURE_VALUE]
            yes_result_dict[curr_feat_val] = count_yes / (len(tags["yes"]) + len(pos_att))
            count_no = 0
            for t in tags["no"]:
                if example[i][ut.FEATURE_VALUE] == t[i][ut.FEATURE_VALUE]:
                    count_no += 1
            curr_feat = example[i][ut.FEATURE_VALUE]
            no_result_dict[curr_feat] = count_no / (len(tags["no"]) + len(pos_att))
        yes = self.calc_prop(yes_result_dict) * Yes_Prob
        no = self.calc_prop(no_result_dict) * No_Prob
        if yes >= no:
            return ut.ret_spec_class('yes')
        else:
            return ut.ret_spec_class('no')
    """
    calc_prop method.
    calculate the full propbaility of a given classificatin
    by multiply all the sub-propabilities.
    """
    def calc_prop(self, _dict):
        values = _dict.values()
        mult = 1
        for val in values:
            mult *= val
        return mult

