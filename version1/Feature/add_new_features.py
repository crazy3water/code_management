import numpy as np

def del_buied(someday):
    try:
        buied_label_ = someday['build_label']
        befor_buied = [someday['behavior_type'] == 4]
        dict = {True: 1, False: 0}
        buied_label = list(map(lambda x: dict[x], befor_buied))
        buied_label = np.sum(buied_label_,buied_label)
    except:
        befor_buied = [someday['behavior_type'] == 4]
        dict = {True:1 , False:0}
        buied_label = list(map(lambda x:dict[x],befor_buied))
    someday['build_label'] = buied_label
    return someday
