FEATURE_TYPE = 0
FEATURE_VALUE = 1

Features_option = {}
Yes_Prob = None
No_Prob = None
Classification_length = None

"""
init_features_dict method.
initiliaze a dictionary between feature type and
feature value.
"""
Pos = None
Neg = None
def init_class(name):
    global Pos, Neg
    if name in ['yes', 'T', 'True','true', "yes", "T", "True", "true"]:
        if Pos == None:
            Pos = name
            return
    elif name in ['no', 'F', 'False','false', "no", "F", "False", "false"]:
        if Neg == None:
            Neg = name
            return
def ret_spec_class(name):
    if name in ['yes', 'T', 'True']:
        return Pos
    elif name in ['no', 'F', 'False']:
        return Neg

def init_features_dict(props, examples):
    global Features_option
    i = 0
    for feat in props:
        options_set = {'d'}
        options_set.remove('d')
        for ex in examples:
            options_set.add(ex[i][FEATURE_VALUE])
        i += 1
        Features_option[feat] = set_to_set(options_set)
        options_set.clear()

"""
set_to_set method.
move an entire set to another set
"""
def set_to_set(options):
    new_set = {options.pop()}
    for en in options:
        new_set.add(en)
    return new_set