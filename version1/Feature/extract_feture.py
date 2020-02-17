import pandas as pd
import numpy as np
import datetime
import sys
import time
import xgboost as xgb
from Feature.add_feture import *
from Feature.add_new_features import *
FEATURE_EXTRACTION_SLOT = 10
LabelDay = datetime.datetime(2014,12,18,0,0,0)
BASE_DIR = r'G:\command_study\tianchi1/'
Data = pd.read_csv(BASE_DIR + "DataSet/drop1112_sub_item.csv")
Data['daystime'] = Data['days'].map(lambda x: time.strptime(x, "%Y-%m-%d")).map(lambda x: datetime.datetime(*x[:6]))

#get_train(Data, LabelDay)
def get_train(train_user,end_time):
    # 取出label day 前一天的记录作为打标记录
    data_train = train_user[(train_user['daystime'] == (end_time-datetime.timedelta(days=1)))]#&((train_user.behavior_type==3)|(train_user.behavior_type==2))
    # 训练样本中，删除重复的样本
    data_train = data_train.drop_duplicates(['user_id', 'item_id'])
    data_train_ui = data_train['user_id'] / data_train['item_id']
#    print(len(data_train))

    # 使用label day 的实际购买情况进行打标
    data_label = train_user[train_user['daystime'] == end_time]
    data_label_buy = data_label[data_label['behavior_type'] == 4]
    data_label_buy_ui = data_label_buy['user_id'] / data_label_buy['item_id']

    # 对前一天的交互记录进行打标
    #找到 前一天有交互记录，并且第二天进行了购买的用户
    data_train_labeled = data_train_ui.isin(data_label_buy_ui)
    dict = {True: 1, False: 0}
    data_train_labeled = data_train_labeled.map(dict)

    data_train['label'] = data_train_labeled
    return data_train[['user_id', 'item_id','item_category', 'label']]

def get_label_testset(train_user,LabelDay):
    # 测试集选为上一天所有的交互数据
    data_test = train_user[(train_user['daystime'] == LabelDay)]#&((train_user.behavior_type==3)|(train_user.behavior_type==2))
    data_test = data_test.drop_duplicates(['user_id', 'item_id'])
    return data_test[['user_id', 'item_id','item_category']]



def item_category_feture(data,end_time,beforeoneday):
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    item_count = pd.crosstab(data.item_category,data.behavior_type)
    item_count_before5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        item_count_before5 = pd.crosstab(beforefiveday.item_category,beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        item_count_before5 = pd.crosstab(beforefiveday.item_category,beforefiveday.behavior_type)
    item_count_before_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)

    item_count_before_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
        
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    beforeonedayitem_count = pd.crosstab(beforeoneday.item_category,beforeoneday.behavior_type)
    countAverage = item_count/FEATURE_EXTRACTION_SLOT
    buyRate = pd.DataFrame()
    buyRate['click'] = item_count[1]/item_count[4]
    buyRate['skim'] = item_count[2]/item_count[4]
    buyRate['collect'] = item_count[3]/item_count[4]
    buyRate.index = item_count.index

    buyRate_2 = pd.DataFrame()
    buyRate_2['click'] = item_count_before5[1]/item_count_before5[4]
    buyRate_2['skim'] = item_count_before5[2]/item_count_before5[4]
    buyRate_2['collect'] = item_count_before5[3]/item_count_before5[4]
    buyRate_2.index = item_count_before5.index

    buyRate_3 = pd.DataFrame()
    buyRate_3['click'] = item_count_before_3[1]/item_count_before_3[4]
    buyRate_3['skim'] = item_count_before_3[2]/item_count_before_3[4]
    buyRate_3['collect'] = item_count_before_3[3]/item_count_before_3[4]
    buyRate_3.index = item_count_before_3.index


    buyRate = buyRate.replace([np.inf, -np.inf], 0)
    buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
    buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)
    item_category_feture = pd.merge(item_count,beforeonedayitem_count,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,countAverage,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,buyRate,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,item_count_before5,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,item_count_before_3,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,item_count_before_2,how='left',right_index=True,left_index=True)
#    item_category_feture = pd.merge(item_category_feture,buyRate_2,how='left',right_index=True,left_index=True)
#    item_category_feture = pd.merge(item_category_feture,buyRate_3,how='left',right_index=True,left_index=True)
    item_category_feture.fillna(0,inplace=True)
    return item_category_feture

def item_id_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    item_count = pd.crosstab(data.item_id,data.behavior_type)
    item_count_before5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        item_count_before5 = pd.crosstab(beforefiveday.item_id,beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        item_count_before5 = pd.crosstab(beforefiveday.item_id,beforefiveday.behavior_type)

    item_count_before_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)

    item_count_before_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)
        
    item_count_unq = data.groupby(by = ['item_id','behavior_type']).agg({"user_id":lambda x:x.nunique()})
    item_count_unq = item_count_unq.unstack()
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    beforeonedayitem_count = pd.crosstab(beforeoneday.item_id,beforeoneday.behavior_type)
    countAverage = item_count/FEATURE_EXTRACTION_SLOT
    buyRate = pd.DataFrame()
    buyRate['click'] = item_count[1]/item_count[4]
    buyRate['skim'] = item_count[2]/item_count[4]
    buyRate['collect'] = item_count[3]/item_count[4]
    buyRate.index = item_count.index

    buyRate_2 = pd.DataFrame()
    buyRate_2['click'] = item_count_before5[1]/item_count_before5[4]
    buyRate_2['skim'] = item_count_before5[2]/item_count_before5[4]
    buyRate_2['collect'] = item_count_before5[3]/item_count_before5[4]
    buyRate_2.index = item_count_before5.index

    buyRate_3 = pd.DataFrame()
    buyRate_3['click'] = item_count_before_3[1]/item_count_before_3[4]
    buyRate_3['skim'] = item_count_before_3[2]/item_count_before_3[4]
    buyRate_3['collect'] = item_count_before_3[3]/item_count_before_3[4]
    buyRate_3.index = item_count_before_3.index

    buyRate = buyRate.replace([np.inf, -np.inf], 0)
    buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
    buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)
    item_id_feture = pd.merge(item_count,beforeonedayitem_count,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,countAverage,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,buyRate,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,item_count_unq,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,item_count_before5,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,item_count_before_3,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,item_count_before_2,how='left',right_index=True,left_index=True)
#    item_id_feture = pd.merge(item_id_feture,buyRate_2,how='left',right_index=True,left_index=True)
#    item_id_feture = pd.merge(item_id_feture,buyRate_3,how='left',right_index=True,left_index=True)
    item_id_feture.fillna(0,inplace=True)
    return item_id_feture


def user_id_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    user_count = pd.crosstab(data.user_id,data.behavior_type)
    user_count_before5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        user_count_before5 = pd.crosstab(beforefiveday.user_id,beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        user_count_before5 = pd.crosstab(beforefiveday.user_id,beforefiveday.behavior_type)

    user_count_before_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        user_count_before_3 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        user_count_before_3 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)

    user_count_before_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        user_count_before_2 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        user_count_before_2 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)
        
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    beforeonedayuser_count = pd.crosstab(beforeoneday.user_id,beforeoneday.behavior_type)
    countAverage = user_count/FEATURE_EXTRACTION_SLOT
    buyRate = pd.DataFrame()
    buyRate['click'] = user_count[1]/user_count[4]
    buyRate['skim'] = user_count[2]/user_count[4]
    buyRate['collect'] = user_count[3]/user_count[4]
    buyRate.index = user_count.index

    buyRate_2 = pd.DataFrame()
    buyRate_2['click'] = user_count_before5[1]/user_count_before5[4]
    buyRate_2['skim'] = user_count_before5[2]/user_count_before5[4]
    buyRate_2['collect'] = user_count_before5[3]/user_count_before5[4]
    buyRate_2.index = user_count_before5.index

    buyRate_3 = pd.DataFrame()
    buyRate_3['click'] = user_count_before_3[1]/user_count_before_3[4]
    buyRate_3['skim'] = user_count_before_3[2]/user_count_before_3[4]
    buyRate_3['collect'] = user_count_before_3[3]/user_count_before_3[4]
    buyRate_3.index = user_count_before_3.index


    buyRate = buyRate.replace([np.inf, -np.inf], 0)
    buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
    buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)
    #tpt最大最小差
    long_online = pd.pivot_table(beforeoneday,index=['user_id'],values=['hours'],aggfunc=[np.min,np.max,np.ptp])


    user_id_feture = pd.merge(user_count,beforeonedayuser_count,how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,countAverage,how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,buyRate,how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,user_count_before5,how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,user_count_before_3,how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,user_count_before_2,how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,long_online,how='left',right_index=True,left_index=True)
#    user_id_feture = pd.merge(user_id_feture,buyRate_2,how='left',right_index=True,left_index=True)
#    user_id_feture = pd.merge(user_id_feture,buyRate_3,how='left',right_index=True,left_index=True)
    user_id_feture.fillna(0,inplace=True)
    return user_id_feture



def user_item_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    user_item_count = pd.crosstab([data.user_id,data.item_id],data.behavior_type)
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    user_item_count_5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        user_item_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_id],beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        user_item_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_id],beforefiveday.behavior_type)
    user_item_count_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        user_item_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        user_item_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)

    user_item_count_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        user_item_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        user_item_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)
        
    beforeonedayuser_item_count = pd.crosstab([beforeoneday.user_id,beforeoneday.item_id],beforeoneday.behavior_type)
    
#    _live = user_item_long_touch(data)
    
    
    max_touchtime = pd.pivot_table(beforeoneday,index=['user_id','item_id'],values=['hours'],aggfunc=[np.min,np.max])
    max_touchtype = pd.pivot_table(beforeoneday,index=['user_id','item_id'],values=['behavior_type'],aggfunc=np.max)
    user_item_feture = pd.merge(user_item_count,beforeonedayuser_item_count,how='left',right_index=True,left_index=True)
    user_item_feture = pd.merge(user_item_feture,max_touchtime,how='left',right_index=True,left_index=True)
    user_item_feture = pd.merge(user_item_feture,max_touchtype,how='left',right_index=True,left_index=True)
#    user_item_feture = pd.merge(user_item_feture,_live,how='left',right_index=True,left_index=True)

    user_item_feture = pd.merge(user_item_feture,user_item_count_5,how='left',right_index=True,left_index=True)
    user_item_feture = pd.merge(user_item_feture,user_item_count_3,how='left',right_index=True,left_index=True)
    user_item_feture = pd.merge(user_item_feture,user_item_count_2,how='left',right_index=True,left_index=True)
    user_item_feture.fillna(0,inplace=True)
    return user_item_feture

def user_cate_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    user_item_count = pd.crosstab([data.user_id,data.item_category],data.behavior_type)
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    
    user_cate_count_5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=(end_time-datetime.timedelta(days=5+2))]
        user_cate_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_category],beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=(end_time-datetime.timedelta(days=5))]
        user_cate_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_category],beforefiveday.behavior_type)
    user_cate_count_3 = None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=3+2))]
        user_cate_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=3))]
        user_cate_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)


    user_cate_count_2 = None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=7+2))]
        user_cate_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=7))]
        user_cate_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)
        
#    _live = user_cate_long_touch(data)
    beforeonedayuser_item_count = pd.crosstab([beforeoneday.user_id,beforeoneday.item_category],beforeoneday.behavior_type)
    max_touchtime = pd.pivot_table(beforeoneday,index=['user_id','item_category'],values=['hours'],aggfunc=[np.min,np.max])
    max_touchtype = pd.pivot_table(beforeoneday,index=['user_id','item_category'],values=['behavior_type'],aggfunc=np.max)
    user_cate_feture = pd.merge(user_item_count,beforeonedayuser_item_count,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,max_touchtime,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,max_touchtype,how='left',right_index=True,left_index=True)
#    user_cate_feture = pd.merge(user_cate_feture,_live,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,user_cate_count_5,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,user_cate_count_3,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,user_cate_count_2,how='left',right_index=True,left_index=True)
    user_cate_feture.fillna(0,inplace=True)
    return user_cate_feture


if __name__ == '__main__':
    result=[]
    for i in range(15):
        train_user_window1 = None
        # LabelDay: 12-18
        if (LabelDay >= datetime.datetime(2014,12,12,0,0,0)):
            # train_user_window1数据范围：12-06 ~ 12-18
            train_user_window1 = Data[(Data['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT+2))) & (Data['daystime'] < LabelDay)]
        else:
            train_user_window1 = Data[(Data['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))) & (Data['daystime'] < LabelDay)]
#        train_user_window1 = Data[(Data['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))) & (Data['daystime'] < LabelDay)]
        beforeoneday = Data[Data['daystime'] == (LabelDay-datetime.timedelta(days=1))]

        # beforetwoday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=2))) & (Data['daystime'] < LabelDay)]
        # beforefiveday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=5))) & (Data['daystime'] < LabelDay)]
        #得到 x 的训练集，其中将 所有的用户的行为与labelDay的购买行为进行标签：如果购买了则为true，否则为false
        x = get_train(Data, LabelDay)
        #新的特征
        beforeoneday = del_buied(beforeoneday)

        add_user_click_1 = user_click(beforeoneday)
        add_user_item_click_1 = user_item_click(beforeoneday)
        add_user_cate_click_1 = user_cate_click(beforeoneday)
        # add_user_click_2 = user_click(beforetwoday)
        # add_user_click_5 = user_click(beforefiveday)
        liveday = user_liveday(train_user_window1)
        # sys.exit()
        a = user_id_feture(train_user_window1, LabelDay,beforeoneday)
        a = a.reset_index()
        b = item_id_feture(train_user_window1, LabelDay,beforeoneday)
        b = b.reset_index()
        c = item_category_feture(train_user_window1, LabelDay,beforeoneday)
        c = c.reset_index()
        d = user_cate_feture(train_user_window1, LabelDay,beforeoneday)
        d = d.reset_index()
        e = user_item_feture(train_user_window1, LabelDay,beforeoneday)
        e = e.reset_index()
        x = pd.merge(x,a,on=['user_id'],how='left')
        x = pd.merge(x,b,on=['item_id'],how='left')
        x = pd.merge(x,c,on=['item_category'],how='left')
        x = pd.merge(x,d,on=['user_id','item_category'],how='left')
        x = pd.merge(x,e,on=['user_id','item_id'],how='left')
        x = pd.merge(x,add_user_click_1,left_on = ['user_id'],right_index=True,how = 'left' )
        # x = pd.merge(x,add_user_click_2,left_on = ['user_id'],right_index=True,how = 'left' )
        # x = pd.merge(x,add_user_click_5,left_on = ['user_id'],right_index=True,how = 'left' )
        x = pd.merge(x,add_user_item_click_1,left_on = ['user_id','item_id'],right_index=True,how = 'left' )
        x = pd.merge(x,add_user_cate_click_1,left_on = ['user_id','item_category'],right_index=True,how = 'left' )
        x = pd.merge(x,liveday,left_on = ['user_id'],right_index=True,how = 'left' )
        x = x.fillna(0)
        print(i,LabelDay,len(x))
        LabelDay = LabelDay-datetime.timedelta(days=1)
        if (LabelDay == datetime.datetime(2014,12,13,0,0,0)):
            LabelDay = datetime.datetime(2014,12,10,0,0,0)
        result.append(x)
    train_set = pd.concat(result,axis=0,ignore_index=True)
    train_set.to_csv(r'../result/train_train_no_jiagou.csv',index=None)
    ###############################################
    
    LabelDay=datetime.datetime(2014,12,18,0,0,0)
    test = get_label_testset(Data,LabelDay)

    train_user_window1 =  Data[(Data['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT-1))) & (Data['daystime'] <= LabelDay)]
    beforeoneday = Data[Data['daystime'] == LabelDay]
    beforeoneday = del_buied(beforeoneday)
    # beforetwoday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=2))) & (Data['daystime'] < LabelDay)]
    # beforefiveday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=5))) & (Data['daystime'] < LabelDay)]
    add_user_click = user_click(beforeoneday)
    add_user_item_click = user_item_click(beforeoneday)
    add_user_cate_click = user_cate_click(beforeoneday)
    # add_user_click_2 = user_click(beforetwoday)
    # add_user_click_5 = user_click(beforefiveday)
    liveday = user_liveday(train_user_window1)
    a = user_id_feture(train_user_window1, LabelDay,beforeoneday)
    a = a.reset_index()
    b = item_id_feture(train_user_window1, LabelDay,beforeoneday)
    b = b.reset_index()
    c = item_category_feture(train_user_window1, LabelDay,beforeoneday)
    c = c.reset_index()
    d = user_cate_feture(train_user_window1, LabelDay,beforeoneday)
    d = d.reset_index()
    e = user_item_feture(train_user_window1, LabelDay,beforeoneday)
    e = e.reset_index()
    test = pd.merge(test,a,on=['user_id'],how='left')
    test = pd.merge(test,b,on=['item_id'],how='left')
    test = pd.merge(test,c,on=['item_category'],how='left')
    test = pd.merge(test,d,on=['user_id','item_category'],how='left')
    test = pd.merge(test,e,on=['user_id','item_id'],how='left')
    test = pd.merge(test,add_user_click,left_on = ['user_id'],right_index=True,how = 'left' )
    # test = pd.merge(test,add_user_click_2,left_on = ['user_id'],right_index=True,how = 'left' )
    # test = pd.merge(test,add_user_click_5,left_on = ['user_id'],right_index=True,how = 'left' )
    test = pd.merge(test,add_user_item_click,left_on = ['user_id','item_id'],right_index=True,how = 'left' )
    test = pd.merge(test,add_user_cate_click,left_on = ['user_id','item_category'],right_index=True,how = 'left' )
    test = pd.merge(test,liveday,left_on = ['user_id'],right_index=True,how = 'left' )
    test = test.fillna(0)
    test.to_csv(r'../result/test_test_no_jiagou.csv',index=None)