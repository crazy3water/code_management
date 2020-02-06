import pandas as pd
import numpy as np
BASE_DIR = r'G:\command_study\tianchi1/'

if __name__ == '__main__':
	user_table = pd.read_csv(BASE_DIR + 'DataSet/tianchi_fresh_comp_train_user.csv')
	item_table = pd.read_csv(BASE_DIR + 'DataSet/tianchi_fresh_comp_train_item.csv')
	#提取 user中商品 包含了哪些 item商品
	user_table = user_table[user_table.item_id.isin(list(item_table.item_id))]
	user_table['days'] = user_table['time'].map(lambda x:x.split(' ')[0])
	user_table['hours'] = user_table['time'].map(lambda x:x.split(' ')[1])
	user_table = user_table[user_table['days'] != '2014-12-12']
	user_table = user_table[user_table['days'] != '2014-12-11']
	user_table.to_csv(BASE_DIR + 'DataSet/drop1112_sub_item.csv',index=None)

