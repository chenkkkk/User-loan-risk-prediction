# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 01:35:22 2018

@author: CCL
"""

import pandas as pd
import datetime
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import xgboost as xgb
import re
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc


def xgb_valid(train_set_x,train_set_y):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'rank:pairwise',
              'eval_metric' : 'auc',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'silent':1,
              'nthread':8
              }
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    model = xgb.cv(params, dtrain, num_boost_round=1000,nfold=5,metrics={'auc'},seed=10)
    print(model)


def xgb_feature(train_set_x,train_set_y,test_set_x,test_set_y):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'rank:pairwise',
              'eval_metric' : 'auc',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'silent':1
              }
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)
    model = xgb.train(params, dtrain, num_boost_round=800)
    predict = model.predict(dvali)
    return predict

IS_OFFLine = False
if __name__ == '__main__':
    #%%认证表特征
    train_auth = pd.read_csv('../AI_risk_train_V3.0/train_auth_info.csv',parse_dates = ['auth_time'])
    #注册时是否有时间
    train_auth['is_auth_time_authtable'] = train_auth['auth_time'].map(lambda x:0 if str(x)=='nan' else 1)
    #注册时是否有idcard
    train_auth['is_idcard_authtable'] = train_auth['id_card'].map(lambda x:0 if str(x)=='nan' else 1)

    #注册时是否有phone
    train_auth['is_phone_authtable'] = train_auth['phone'].map(lambda x:0 if str(x)=='nan' else 1)


    #%%银行卡特征
    train_bankcard = pd.read_csv('../AI_risk_train_V3.0/train_bankcard_info.csv')
    bank_name_setlen = train_bankcard.groupby(by= ['id'], as_index= False)['bank_name'].agg({'bank_name_len':lambda x:len(set(x))})
    bank_num_len = train_bankcard.groupby(by= ['id'], as_index = False)['tail_num'].agg({'tail_num_len':lambda x:len(x)})
    bank_phone_num_setlen = train_bankcard.groupby(by= ['id'], as_index = False)['phone'].agg({'bank_phone_num':lambda x:x.nunique()})
    
    train_bankcard['card_type_score'] = train_bankcard['card_type'].map(lambda x:0.0154925 if x=='信用卡' else 0.02607069)
    bank_card_type_score = train_bankcard.groupby(by= ['id'], as_index = False)['card_type_score'].agg({'card_type_score_mean':np.mean})
 
    #%%信誉表特征
    train_credit = pd.read_csv('../AI_risk_train_V3.0/train_credit_info.csv')
    #额度-使用值
    train_credit['can_use_credittable'] = train_credit['quota'] - train_credit['overdraft']

    #%%订单表特征
    train_order = pd.read_csv('../AI_risk_train_V3.0/train_order_info.csv',parse_dates=['time_order'])
    train_order['amt_order_ordertable'] = train_order['amt_order'].map(lambda x:np.nan if ((x == 'NA')| (x == 'null')) else float(x))
    train_order['unit_price_ordertable'] = train_order['unit_price'].map(lambda x:np.nan if ((x == 'NA')| (x == 'null')) else float(x))
    
    train_order['time_order_ordertable'] = train_order['time_order'].map(lambda x : pd.lib.NaT if (str(x) == '0' or x == 'NA' or x == 'nan')
                                else (datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S') if ':' in str(x)
                                else (datetime.datetime.utcfromtimestamp(int(x[0:10])) + datetime.timedelta(hours = 8))))






    #%%收货地址特征
    train_recieve = pd.read_csv('../AI_risk_train_V3.0/train_recieve_addr_info.csv')
    train_recieve['first_name'] = train_recieve['region'].map(lambda x:str(x)[:2])
    train_recieve['last_name'] = train_recieve['region'].map(lambda x:str(x)[-1])


    #%%target表特征
    train_target = pd.read_csv('../AI_risk_train_V3.0/train_target.csv',parse_dates = ['appl_sbm_tm'])


    #%%用户表特征
    train_user = pd.read_csv('../AI_risk_train_V3.0/train_user_info.csv')
    train_user = train_user.drop(['merriage','income','id_card','degree','industry'],axis=1)
    # train_user = train_user.drop(['id_card'],axis=1)
    train_user['is_hobby_usertable'] = train_user['hobby'].map(lambda x:0 if str(x)=='nan' else 1)
    train_user['is_birthday_usertable'] = train_user['birthday'].map(lambda x:0 if str(x)=='nan' else 1)
    train_user['birthday'] = train_user['birthday'].map(lambda x:datetime.datetime.strptime(str(x),'%Y-%m-%d') if(re.match('19\d{2}-\d{1,2}-\d{1,2}',str(x)) and '-0' not in str(x)) else pd.lib.NaT)




    train_data = pd.merge(train_target,train_auth,on=['id'],how='left')
    train_data = pd.merge(train_data,train_user,on=['id'],how='left')
    train_data = pd.merge(train_data,train_credit,on=['id'],how='left')
    train_data['loan_hour'] = train_data['appl_sbm_tm'].map(lambda x:x.hour)
    train_data['loan_day'] = train_data['appl_sbm_tm'].map(lambda x:x.day)
    train_data['loan_month'] = train_data['appl_sbm_tm'].map(lambda x:x.month)
    train_data['loan_year'] = train_data['appl_sbm_tm'].map(lambda x:x.year)
    train_data['nan_num'] = train_data.isnull().sum(axis=1)
    train_data['diff_day'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['auth_time']).days,axis=1)
    train_data['how_old'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['birthday']).days,axis=1)
    train_data['是否认证时间在借贷时间前'] = train_data.apply(lambda x:0 if (x['is_auth_time_authtable'] == 0) else ( 1 if x['auth_time'] < x['appl_sbm_tm'] else 0),axis=1)
    train_data['是否认证时间在借贷时间后'] = train_data.apply(lambda x:0 if (x['is_auth_time_authtable'] == 0) else ( 1 if x['auth_time'] > x['appl_sbm_tm'] else 0),axis=1)
    train_data['认证时间在借贷时间前多少天'] = train_data.apply(lambda x:0 if (x['是否认证时间在借贷时间前'] == 0) else (x['appl_sbm_tm'] - x['auth_time']).days,axis=1)
    train_data['认证时间在借贷时间后多少天'] = train_data.apply(lambda x:0 if (x['是否认证时间在借贷时间后'] == 0) else (x['auth_time'] - x['appl_sbm_tm']).days,axis=1)
    train_data = pd.merge(train_data,bank_name_setlen,on=['id'],how='left')
    train_data = pd.merge(train_data,bank_num_len,on=['id'],how='left')
    train_data = pd.merge(train_data,bank_phone_num_setlen,on=['id'],how='left')
    train_data = pd.merge(train_data,bank_card_type_score,on=['id'],how='left')

    #%%为订单表建立临时表1
    tmp_train_order = pd.merge(train_order, train_target, on = ['id'])
    tmp_train_order_before_appl_sbm_tm = tmp_train_order[tmp_train_order.time_order_ordertable < tmp_train_order.appl_sbm_tm]
    tmp_train_order_after_appl_sbm_tm = tmp_train_order[tmp_train_order.time_order_ordertable > tmp_train_order.appl_sbm_tm]
    before_appl_sbm_tm_howmany = tmp_train_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间前有多少次购买':len})
    after_appl_sbm_tm_howmany = tmp_train_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间后有多少次购买':len})
    before_appl_sbm_tm_money_mean = tmp_train_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间前有多少次购买':np.mean})
    after_appl_sbm_tm_money_mean = tmp_train_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间后有多少次购买':np.mean})
    before_appl_sbm_tm_money_max = tmp_train_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间前有多少次购买最大值':np.max})
    after_appl_sbm_tm_money_min = tmp_train_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间后有多少次购买最小值':np.min})


    # before_appl_sbm_tm_howmany_unitprice = tmp_train_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间前有多少次购买unit_price':len})
    # after_appl_sbm_tm_howmany_unitprice = tmp_train_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间后有多少次购买unit_price':len})
    # before_appl_sbm_tm_money_mean_unitprice = tmp_train_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间前有多少次购买unit_price':np.mean})
    # after_appl_sbm_tm_money_mean_unitprice = tmp_train_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间后有多少次购买unit_price':np.mean})
    # before_appl_sbm_tm_money_max_unitprice = tmp_train_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间前有多少次购买最大值unit_price':np.max})
    # after_appl_sbm_tm_money_min_unitprice = tmp_train_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间后有多少次购买最小值unit_price':np.min})


    ##建立银行违约率临时表
    tmp_bank_target = pd.merge(train_bankcard,train_target,on=['id'])
    ccc = pd.crosstab(tmp_bank_target.bank_name, tmp_bank_target.target)
    ccc['违约率'] = ccc[1] / (ccc[0]+0.1)
    ccc.reset_index(inplace=True)
    tmp_bank_target = pd.merge(tmp_bank_target, ccc,on = ['bank_name'],how='left')
    bank_name_score_mean = tmp_bank_target.groupby(by= ['id'], as_index = False)['违约率'].agg({'违约率_mean':np.mean})

    train_data = pd.merge(train_data,bank_name_score_mean,on=['id'],how='left')


    ###建立收货地违约率临时表
    # tmp_recieve_target = pd.merge(train_recieve,train_target,on=['id'])
    # ccc = pd.crosstab(tmp_recieve_target.first_name, tmp_recieve_target.target)
    # ccc['recieve违约率'] = ccc[1] / (ccc[0]+0.1)
    # ccc.reset_index(inplace=True)
    # tmp_recieve_target = pd.merge(tmp_recieve_target, ccc,on = ['first_name'],how='left')
    # recieve_score_mean = tmp_recieve_target.groupby(by= ['id'], as_index = False)['recieve违约率'].agg({'recieve违约率_mean':np.mean})
    # train_data = pd.merge(train_data,recieve_score_mean,on=['id'],how='left')


    train_data = pd.merge(train_data,before_appl_sbm_tm_howmany,on=['id'],how='left')
    train_data = pd.merge(train_data,after_appl_sbm_tm_howmany,on=['id'],how='left')
    train_data = pd.merge(train_data,before_appl_sbm_tm_money_mean,on=['id'],how='left')
    train_data = pd.merge(train_data,after_appl_sbm_tm_money_mean,on=['id'],how='left')
    train_data = pd.merge(train_data,before_appl_sbm_tm_money_max,on=['id'],how='left')
    train_data = pd.merge(train_data,after_appl_sbm_tm_money_min,on=['id'],how='left')



    # train_data = pd.merge(train_data,before_appl_sbm_tm_howmany_unitprice,on=['id'],how='left')
    # train_data = pd.merge(train_data,after_appl_sbm_tm_howmany_unitprice,on=['id'],how='left')
    # train_data = pd.merge(train_data,before_appl_sbm_tm_money_mean_unitprice,on=['id'],how='left')
    # train_data = pd.merge(train_data,after_appl_sbm_tm_money_mean_unitprice,on=['id'],how='left')
    # train_data = pd.merge(train_data,before_appl_sbm_tm_money_max_unitprice,on=['id'],how='left')
    # train_data = pd.merge(train_data,after_appl_sbm_tm_money_min_unitprice,on=['id'],how='left')




    train_data = train_data.fillna(0)
    if IS_OFFLine == False:
        train_data = train_data[train_data.appl_sbm_tm >= datetime.datetime(2017,1,1)]
        train_data = train_data.drop(['appl_sbm_tm','id','auth_time','phone','birthday','hobby','id_card'],axis=1)
        print(train_data.shape)
    
    if IS_OFFLine == True:
        dummy_fea = ['sex', 'qq_bound', 'wechat_bound','account_grade']
        dummy_df = pd.get_dummies(train_data.loc[:,dummy_fea])
        train_data_copy = pd.concat([train_data,dummy_df],axis=1)
        train_data_copy = train_data_copy.fillna(0)
        vaild_train_data = train_data_copy.drop(dummy_fea,axis=1)
        valid_train_train = vaild_train_data[vaild_train_data.appl_sbm_tm < datetime.datetime(2017,4,1)]
        valid_train_test = vaild_train_data[vaild_train_data.appl_sbm_tm >= datetime.datetime(2017,4,1)]
        valid_train_train = valid_train_train.drop(['appl_sbm_tm','id','auth_time','phone','birthday','hobby','id_card'],axis=1)
        valid_train_test = valid_train_test.drop(['appl_sbm_tm','id','auth_time','phone','birthday','hobby','id_card'],axis=1)
        vaild_train_x = valid_train_train.drop(['target'],axis=1)
        vaild_test_x = valid_train_test.drop(['target'],axis=1)
        redict_result = xgb_feature(vaild_train_x,valid_train_train['target'].values,vaild_test_x,None)
        print('valid auc',roc_auc_score(valid_train_test['target'].values,redict_result))
        sys.exit(23)

 
    #%%认证表特征
    test_auth = pd.read_csv('../AI_risk_test_V3.0/test_auth_info.csv',parse_dates = ['auth_time'])
    #注册时是否有时间
    test_auth['is_auth_time_authtable'] = test_auth['auth_time'].map(lambda x:0 if ((str(x)=='nan')|(str(x)=='0000-00-00'))  else 1)
    #注册时是否有idcard
    test_auth['is_idcard_authtable'] = test_auth['id_card'].map(lambda x:0 if str(x)=='nan' else 1)

    #注册时是否有phone
    test_auth['is_phone_authtable'] = test_auth['phone'].map(lambda x:0 if str(x)=='nan' else 1)
    test_auth['auth_time'].replace('0000-00-00','nan',inplace=True)
    test_auth['auth_time'] = pd.to_datetime(test_auth['auth_time'])


    #%%银行卡特征
    test_bankcard = pd.read_csv('../AI_risk_test_V3.0/test_bankcard_info.csv')
    bank_name_setlen = test_bankcard.groupby(by= ['id'], as_index= False)['bank_name'].agg({'bank_name_len':lambda x:len(set(x))})
    bank_num_len = test_bankcard.groupby(by= ['id'], as_index = False)['tail_num'].agg({'tail_num_len':lambda x:len(x)})
    bank_phone_num_setlen = test_bankcard.groupby(by= ['id'], as_index = False)['phone'].agg({'bank_phone_num':lambda x:x.nunique()})
    test_bankcard['card_type_score'] = test_bankcard['card_type'].map(lambda x:0.0154925 if x=='信用卡' else 0.02607069)
    bank_card_type_score = test_bankcard.groupby(by= ['id'], as_index = False)['card_type_score'].agg({'card_type_score_mean':np.mean})

    #%%信誉表特征
    test_credit = pd.read_csv('../AI_risk_test_V3.0/test_credit_info.csv')
    #额度-使用值
    test_credit['can_use_credittable'] = test_credit['quota'] - test_credit['overdraft']

    #%%订单表特征
    test_order = pd.read_csv('../AI_risk_test_V3.0/test_order_info.csv',parse_dates=['time_order'])
    test_order['amt_order_ordertable'] = test_order['amt_order'].map(lambda x:np.nan if ((x == 'NA')| (x == 'null')) else float(x))
    test_order['unit_price_ordertable'] = test_order['unit_price'].map(lambda x:np.nan if ((x == 'NA')| (x == 'null')) else float(x))
    test_order['time_order_ordertable'] = test_order['time_order'].map(lambda x : pd.lib.NaT if (str(x) == '0' or x == 'NA' or x == 'nan')
                                else (datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S') if ':' in str(x)
                                else (datetime.datetime.utcfromtimestamp(int(x[0:10])) + datetime.timedelta(hours = 8))))




    #%%收货地址特征
    test_recieve = pd.read_csv('../AI_risk_test_V3.0/test_recieve_addr_info.csv')


    #%%target表特征
    test_target = pd.read_csv('../AI_risk_test_V3.0/test_list.csv',parse_dates = ['appl_sbm_tm'])


    #%%用户表特征
    test_user = pd.read_csv('../AI_risk_test_V3.0/test_user_info.csv')
    test_user = test_user.drop(['merriage','income','id_card','degree','industry'],axis=1)
    # test_user = test_user.drop(['id_card'],axis=1)
    test_user['is_hobby_usertable'] = test_user['hobby'].map(lambda x:0 if str(x)=='nan' else 1)
    test_user['is_birthday_usertable'] = test_user['birthday'].map(lambda x:0 if str(x)=='nan' else 1)
    test_user['birthday'] = test_user['birthday'].map(lambda x:datetime.datetime.strptime(str(x),'%Y-%m-%d') if(re.match('19\d{2}-\d{1,2}-\d{1,2}',str(x)) and '-0' not in str(x)) else pd.lib.NaT)






    test_data = pd.merge(test_target,test_auth,on=['id'],how='left')
    test_data = pd.merge(test_data,test_user,on=['id'],how='left')
    test_data = pd.merge(test_data,test_credit,on=['id'],how='left')
    test_data['loan_hour'] = test_data['appl_sbm_tm'].map(lambda x:x.hour)
    test_data['loan_day'] = test_data['appl_sbm_tm'].map(lambda x:x.day)
    test_data['loan_month'] = test_data['appl_sbm_tm'].map(lambda x:x.month)
    test_data['loan_year'] = test_data['appl_sbm_tm'].map(lambda x:x.year)
    test_data['nan_num'] = test_data.isnull().sum(axis=1)
    test_data['diff_day'] = test_data.apply(lambda row: (row['appl_sbm_tm'] - row['auth_time']).days,axis=1)
    test_data['how_old'] = test_data.apply(lambda row: (row['appl_sbm_tm'] - row['birthday']).days,axis=1)
    test_data['是否认证时间在借贷时间前'] = test_data.apply(lambda x:0 if (x['is_auth_time_authtable'] == 0) else ( 1 if x['auth_time'] < x['appl_sbm_tm'] else 0),axis=1)
    test_data['是否认证时间在借贷时间后'] = test_data.apply(lambda x:0 if (x['is_auth_time_authtable'] == 0) else ( 1 if x['auth_time'] > x['appl_sbm_tm'] else 0),axis=1)
    test_data['认证时间在借贷时间前多少天'] = test_data.apply(lambda x:0 if (x['是否认证时间在借贷时间前'] == 0) else (x['appl_sbm_tm'] - x['auth_time']).days,axis=1)
    test_data['认证时间在借贷时间后多少天'] = test_data.apply(lambda x:0 if (x['是否认证时间在借贷时间后'] == 0) else (x['auth_time'] - x['appl_sbm_tm']).days,axis=1)
    test_data = pd.merge(test_data,bank_name_setlen,on=['id'],how='left')
    test_data = pd.merge(test_data,bank_num_len,on=['id'],how='left')
    test_data = pd.merge(test_data,bank_phone_num_setlen,on=['id'],how='left')
    test_data = pd.merge(test_data,bank_card_type_score,on=['id'],how='left')
 
    #%%为订单表建立临时表
    tmp_test_order = pd.merge(test_order, test_target, on = ['id'])
    tmp_test_order_before_appl_sbm_tm = tmp_test_order[tmp_test_order.time_order_ordertable < tmp_test_order.appl_sbm_tm]
    tmp_test_order_after_appl_sbm_tm = tmp_test_order[tmp_test_order.time_order_ordertable > tmp_test_order.appl_sbm_tm]
    before_appl_sbm_tm_howmany = tmp_test_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间前有多少次购买':len})
    after_appl_sbm_tm_howmany = tmp_test_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间后有多少次购买':len})
    before_appl_sbm_tm_money_mean = tmp_test_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间前有多少次购买':np.mean})
    after_appl_sbm_tm_money_mean = tmp_test_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间后有多少次购买':np.mean})
    before_appl_sbm_tm_money_max = tmp_test_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间前有多少次购买最大值':np.max})
    after_appl_sbm_tm_money_min = tmp_test_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['amt_order_ordertable'].agg({'借贷时间后有多少次购买最小值':np.min})

    # before_appl_sbm_tm_howmany_unitprice = tmp_test_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间前有多少次购买unit_price':len})
    # after_appl_sbm_tm_howmany_unitprice = tmp_test_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间后有多少次购买unit_price':len})
    # before_appl_sbm_tm_money_mean_unitprice = tmp_test_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间前有多少次购买unit_price':np.mean})
    # after_appl_sbm_tm_money_mean_unitprice = tmp_test_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间后有多少次购买unit_price':np.mean})
    # before_appl_sbm_tm_money_max_unitprice = tmp_test_order_before_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间前有多少次购买最大值unit_price':np.max})
    # after_appl_sbm_tm_money_min_unitprice = tmp_test_order_after_appl_sbm_tm.groupby(by=['id'],as_index=False)['unit_price_ordertable'].agg({'借贷时间后有多少次购买最小值unit_price':np.min})


    ###建立银行违约率临时表

    test_data = pd.merge(test_data,bank_name_score_mean,on=['id'],how='left')




    test_data = pd.merge(test_data,before_appl_sbm_tm_howmany,on=['id'],how='left')
    test_data = pd.merge(test_data,after_appl_sbm_tm_howmany,on=['id'],how='left')
    test_data = pd.merge(test_data,before_appl_sbm_tm_money_mean,on=['id'],how='left')
    test_data = pd.merge(test_data,after_appl_sbm_tm_money_mean,on=['id'],how='left')
    test_data = pd.merge(test_data,before_appl_sbm_tm_money_max,on=['id'],how='left')
    test_data = pd.merge(test_data,after_appl_sbm_tm_money_min,on=['id'],how='left')

    # test_data = pd.merge(test_data,before_appl_sbm_tm_howmany_unitprice,on=['id'],how='left')
    # test_data = pd.merge(test_data,after_appl_sbm_tm_howmany_unitprice,on=['id'],how='left')
    # test_data = pd.merge(test_data,before_appl_sbm_tm_money_mean_unitprice,on=['id'],how='left')
    # test_data = pd.merge(test_data,after_appl_sbm_tm_money_mean_unitprice,on=['id'],how='left')
    # test_data = pd.merge(test_data,before_appl_sbm_tm_money_max_unitprice,on=['id'],how='left')
    # test_data = pd.merge(test_data,after_appl_sbm_tm_money_min_unitprice,on=['id'],how='left')

    test_data = test_data.drop(['appl_sbm_tm','id','auth_time','phone','birthday','hobby','id_card'],axis=1)
    test_data['target'] = -1


    dummy_fea = ['sex', 'qq_bound', 'wechat_bound','account_grade']
    train_test_data = pd.concat([train_data,test_data],axis=0,ignore_index = True)
    train_test_data = train_test_data.fillna(0)
    dummy_df = pd.get_dummies(train_test_data.loc[:,dummy_fea])

    train_test_data = pd.concat([train_test_data,dummy_df],axis=1)
    train_test_data = train_test_data.drop(dummy_fea,axis=1)
    
    train_train = train_test_data.iloc[:train_data.shape[0],:]
    test_test = train_test_data.iloc[train_data.shape[0]:,:]
    
    
    train_train_x = train_train.drop(['target'],axis=1)
    test_test_x = test_test.drop(['target'],axis=1)


    predict_result = xgb_feature(train_train_x,train_train['target'].values,test_test_x,None)

    ans = pd.read_csv('../AI_risk_test_V3.0/test_list.csv',parse_dates = ['appl_sbm_tm'])
    ans['PROB'] = predict_result
    ans = ans.drop(['appl_sbm_tm'],axis=1)
    minmin, maxmax = min(ans['PROB']),max(ans['PROB'])
    ans['PROB'] = ans['PROB'].map(lambda x:(x-minmin)/(maxmax-minmin))
    ans['PROB'] = ans['PROB'].map(lambda x:'%.4f' % x)
    ans.to_csv('../result/rebuild.csv',index=None)