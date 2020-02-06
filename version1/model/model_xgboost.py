import xgboost as xgb
import pandas as pd
import sys
import time
import datetime

BASE_DIR = r'G:\command_study\tianchi1/'
Data = pd.read_csv(BASE_DIR + "DataSet/drop1112_sub_item.csv")
Data['daystime'] = Data['days'].map(lambda x: time.strptime(x, "%Y-%m-%d")).map(lambda x: datetime.datetime(*x[:6]))

if __name__ == '__main__':
    print('---------------读取数据--------------')
    train_set = pd.read_csv(r'../result/train_train_no_jiagou.csv')
    test = pd.read_csv(r'../result/test_test_no_jiagou.csv')
    train_set_1 = train_set[train_set['label'] == 1]
    train_set_0 = train_set[train_set['label'] == 0]
    # 抽取
    print('---------------数据抽样--------------')
    new_train_set_0 = train_set_0.sample(len(train_set_1) * 90)
    train_set = pd.concat([train_set_1, new_train_set_0], axis=0)
    ###############
    train_y = train_set['label'].values
    train_x = train_set.drop(['user_id', 'item_id', 'item_category', 'label'], axis=1).values
    test_x = test.drop(['user_id', 'item_id', 'item_category'], axis=1).values
    num_round = 900
    params = {'max_depth': 4, 'colsample_bytree': 0.8, 'subsample': 0.8, 'eta': 0.02, 'silent': 1,
              'objective': 'binary:logistic', 'eval_metric ': 'error', 'min_child_weight': 2.5,
              # 'max_delta_step':10,'gamma':0.1,'scale_pos_weight':230/1,
              'seed': 10}  #
    plst = list(params.items())
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    print('---------------训练模型--------------')
    bst = xgb.train(plst, dtrain, num_round)
    print('---------------数据测试--------------')
    predicted_proba = bst.predict(dtest)
    # print(predicted_proba)

    predicted_proba = pd.DataFrame(predicted_proba)
    predicted = pd.concat([test[['user_id', 'item_id']], predicted_proba], axis=1)
    predicted.columns = ['user_id', 'item_id', 'prob']
    # print(predicted)
    predicted = predicted.sort_values('prob', axis=0, ascending=False)
    # print(predicted)
    #    predict1 = predicted.iloc[:650, [0, 1]]
    #    # 保存到文件
    #    predict1.to_csv("../result/10_30_2/650_1B80minchildweight1.8.csv", index=False)

    predict2 = predicted.iloc[:700, [0, 1]]
    # 保存到文件
    predict2.to_csv("../result/tianchi_mobile_recommendation_predict.csv", index=False)

    #    predict3 = predicted.iloc[:750, [0, 1]]
    #    # 保存到文件
    #    predict3.to_csv("../result/10_30_2/750_1B80minchildweight1.8.csv", index=False)
    sys.exit()
    #    evaluate(predicted)


    #####################################################################线下验证部分
    reference = Data[Data['daystime'] == (LabelDay + datetime.timedelta(days=1))]
    reference = reference[reference['behavior_type'] == 4]  # 购买的记录
    reference = reference[['user_id', 'item_id']]  # 获取ui对
    reference = reference.drop_duplicates(['user_id', 'item_id'])  # 去重
    ui = predicted['user_id'] / predicted['item_id']

    predicted = predicted[ui.duplicated() == False]

    predicted_ui = predicted['user_id'] / predicted['item_id']
    reference_ui = reference['user_id'] / reference['item_id']

    is_in = predicted_ui.isin(reference_ui)
    true_positive = predicted[is_in]

    tp = len(true_positive)
    predictedSetCount = len(predicted)
    referenceSetCount = len(reference)

    precision = tp / predictedSetCount
    recall = tp / referenceSetCount

    f_score = 2 * precision * recall / (precision + recall)

    tp = recall * referenceSetCount
    predictedSetCount = tp / precision

    print('%.8f%% %.8f %.8f %.0f %.0f' %
          (f_score * 100, precision, recall, tp, predictedSetCount))