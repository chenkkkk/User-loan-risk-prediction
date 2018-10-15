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
IS_OFFLine = False

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
    return predict,model


    
if __name__ == '__main__':
    #%%
    train_auth = pd.read_csv('../AI_risk_train_V3.0/train_auth_info.csv',parse_dates = ['auth_time'])
#    auth_time = train_auth['auth_time'].map(lambda x:0 if str(x)=='nan' else 1)
#    auth_time_df = pd.DataFrame();auth_time_df['id'] = train_auth['id'];auth_time_df['auth_time_df'] = auth_time
    auth_idcard = train_auth['id_card'].map(lambda x:0 if str(x)=='nan' else 1)
    auth_idcard_df = pd.DataFrame();auth_idcard_df['id'] = train_auth['id'];auth_idcard_df['auth_idcard_df'] = auth_idcard
    auth_phone = train_auth['phone'].map(lambda x:0 if str(x)=='nan' else 1)
    auth_phone_df = pd.DataFrame();auth_phone_df['id'] = train_auth['id'];auth_idcard_df['auth_phone_df'] = auth_phone
    #%%
    train_bankcard = pd.read_csv('../AI_risk_train_V3.0/train_bankcard_info.csv')
    "增加特征"
    train_bankcard_bank_count = train_bankcard.groupby(by=['id'], as_index=False)['bank_name'].agg({'bankcard_count':lambda x :len(x)})
    train_bankcard_card_count = train_bankcard.groupby(by=['id'], as_index=False)['card_type'].agg({'card_type_count':lambda x :len(set(x))})
    train_bankcard_phone_count = train_bankcard.groupby(by=['id'], as_index=False)['phone'].agg({'phone_count':lambda x :len(set(x))})

    #%%
    train_credit = pd.read_csv('../AI_risk_train_V3.0/train_credit_info.csv')
    "增加特征"
    #评分的反序
    train_credit['credit_score_inverse'] = train_credit['credit_score'].map(lambda x :605-x)
    #额度-使用值
    train_credit['can_use'] = train_credit['quota'] - train_credit['overdraft']
    #%%
    train_order = pd.read_csv('../AI_risk_train_V3.0/train_order_info.csv',parse_dates=['time_order'])
    train_order['amt_order'] = train_order['amt_order'].map(lambda x:np.nan if ((x == 'NA')| (x == 'null')) else float(x))
    
    train_order['time_order'] = train_order['time_order'].map(lambda x : pd.lib.NaT if (str(x) == '0' or x == 'NA' or x == 'nan')
                                else (datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S') if ':' in str(x)
                                else (datetime.datetime.utcfromtimestamp(int(x[0:10])) + datetime.timedelta(hours = 8))))
    train_order_time_max = train_order.groupby(by=['id'], as_index=False)['time_order'].agg({'train_order_time_max':lambda x:max(x)})
    train_order_time_min = train_order.groupby(by=['id'], as_index=False)['time_order'].agg({'train_order_time_min':lambda x:min(x)})
    train_order_type_zaixian = train_order.groupby(by=['id']).apply(lambda x:x['type_pay'][(x['type_pay']=='在线支付').values].count()).reset_index(name = 'type_pay_zaixian')
    train_order_type_huodao = train_order.groupby(by=['id']).apply(lambda x:x['type_pay'][(x['type_pay']=='货到付款').values].count()).reset_index(name = 'type_pay_huodao')
#    train_order_mean_unit_price = train_order.groupby(by=['id']).apply(lambda x:np.mean(x['unit_price'])).reset_index(name = 'mean_unit_price')
#    train_order_mean_amt_order = train_order.groupby(by=['id']).apply(lambda x:np.mean(x['amt_order'])).reset_index(name = 'mean_amt_order')
#    train_order_phone_unique = train_order.groupby(by=['id']).apply(lambda x:x['phone'].nunique()).reset_index(name = '_order_phone_unique')
#    train_order_many_success = train_order.groupby(by=['id']).apply(lambda x:x['sts_order'][(x['sts_order']=='完成').values].count()).reset_index(name = '_order_many_success')
#    train_order_many_occuer = train_order.groupby(by=['id']).apply(lambda x:x['sts_order'].count()).reset_index(name = '_order_many_occuer')

    #%%
    
    train_recieve = pd.read_csv('../AI_risk_train_V3.0/train_recieve_addr_info.csv')
    train_recieve['region'] = train_recieve['region'].map(lambda x:str(x)[:2])
    tmp_tmp_recieve = pd.crosstab(train_recieve.id,train_recieve.region);tmp_tmp_recieve = tmp_tmp_recieve.reset_index()
    tmp_tmp_recieve_phone_count = train_recieve.groupby(by=['id']).apply(lambda x:x['fix_phone'].count());tmp_tmp_recieve_phone_count=tmp_tmp_recieve_phone_count.reset_index()
    tmp_tmp_recieve_phone_count_unique = train_recieve.groupby(by=['id']).apply(lambda x:x['fix_phone'].nunique());tmp_tmp_recieve_phone_count_unique=tmp_tmp_recieve_phone_count_unique.reset_index()
    #%%
    train_target = pd.read_csv('../AI_risk_train_V3.0/train_target.csv',parse_dates = ['appl_sbm_tm'])
    train_user = pd.read_csv('../AI_risk_train_V3.0/train_user_info.csv')
    is_hobby = train_user['hobby'].map(lambda x:0 if str(x)=='nan' else 1)
    is_hobby_df = pd.DataFrame();is_hobby_df['id'] = train_user['id'];is_hobby_df['is_hobby'] = is_hobby
    is_idcard = train_user['id_card'].map(lambda x:0 if str(x)=='nan' else 1)
    is_idcard_df = pd.DataFrame();is_idcard_df['id'] = train_user['id'];is_idcard_df['is_hobby'] = is_idcard

    #%%usesr_birthday
    tmp_tmp = train_user[['id','birthday']];tmp_tmp = tmp_tmp.set_index(['id'])
    is_double_ = tmp_tmp['birthday'].map(lambda x:(str(x) == '--')*1).reset_index(name='is_double_')
    is_0_0_0 = tmp_tmp['birthday'].map(lambda x:(str(x) == '0-0-0')*1).reset_index(name='is_0_0_0')
    is_1_1_1 = tmp_tmp['birthday'].map(lambda x:(str(x) == '1-1-1')*1).reset_index(name='is_1_1_1')
    is_0000_00_00 = tmp_tmp['birthday'].map(lambda x:(str(x) == '0000-00-00')*1).reset_index(name='is_0000_00_00')
    is_0001_1_1 = tmp_tmp['birthday'].map(lambda x:(str(x) == '0001-1-1')*1).reset_index(name='is_0001_1_1')
    is_hou_in = tmp_tmp['birthday'].map(lambda x:('后' in str(x))*1).reset_index(name='is_hou_in')
    # is_nan = tmp_tmp['birthday'].map(lambda x:(str(x) == 'nan')*1).reset_index(name='is_nan')
    #%%
    train_user['birthday'] = train_user['birthday'].map(lambda x:datetime.datetime.strptime(str(x),'%Y-%m-%d') if(re.match('19\d{2}-\d{1,2}-\d{1,2}',str(x)) and '-0' not in str(x)) else pd.lib.NaT)
#%%合并    以及 基本特征
    train_data = pd.merge(train_target,train_auth,on=['id'],how='left')
    train_data = pd.merge(train_data,train_user,on=['id'],how='left')
    train_data = pd.merge(train_data,train_credit,on=['id'],how='left')
    train_data['hour'] = train_data['appl_sbm_tm'].map(lambda x:x.hour)
    train_data['month'] = train_data['appl_sbm_tm'].map(lambda x:x.month)
    train_data['year'] = train_data['appl_sbm_tm'].map(lambda x:x.year)
    train_data['quota_use_ratio'] = train_data['overdraft'] / (train_data['quota']+0.01)
    train_data['nan_num'] = train_data.isnull().sum(axis=1)
    train_data['diff_day'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['auth_time']).days,axis=1)
    train_data['how_old'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['birthday']).days/365,axis=1)
    #################################################################
    auth_idcard = list(train_data['id_card_x']);user_idcard = list(train_data['id_card_y'])
    idcard_result = []
    for indexx,uu in enumerate(auth_idcard):
        if (str(auth_idcard[indexx])=='nan') and (str(user_idcard[indexx])=='nan'):
            idcard_result.append(0)
        elif (str(auth_idcard[indexx])!='nan') and (str(user_idcard[indexx])=='nan'):
            idcard_result.append(1)
        elif (str(auth_idcard[indexx])=='nan') and (str(user_idcard[indexx])!='nan'):
            idcard_result.append(2)
        else:
            ttt1 = str(auth_idcard[indexx])[0] + str(auth_idcard[indexx])[-1]
            ttt2 = str(user_idcard[indexx])[0] + str(user_idcard[indexx])[-1]
            if ttt1 == ttt2:
                idcard_result.append(3)
            if ttt1 != ttt2:
                idcard_result.append(4)
    train_data['the_same_id'] = idcard_result



    train_bankcard_phone_list = train_bankcard.groupby(by=['id'])['phone'].apply(lambda x:list(set(x.tolist()))).reset_index(name = 'bank_phone_list')
    train_data = pd.merge(train_data,train_bankcard_phone_list,on=['id'],how='left')
    train_data['exist_phone'] = train_data.apply(lambda x:x['phone'] in x['bank_phone_list'],axis=1)
    train_data['exist_phone'] = train_data['exist_phone']*1
    train_data = train_data.drop(['bank_phone_list'],axis=1)
    #%%bankcard_info
    bank_name = train_bankcard.groupby(by= ['id'], as_index= False)['bank_name'].agg({'bank_name_len':lambda x:len(set(x))})
    bank_num = train_bankcard.groupby(by= ['id'], as_index = False)['tail_num'].agg({'tail_num_len':lambda x:len(set(x))})
    bank_phone_num = train_bankcard.groupby(by= ['id'], as_index = False)['phone'].agg({'bank_phone_num':lambda x:x.nunique()})
    
    
    train_data = pd.merge(train_data,bank_name,on=['id'],how='left')
    train_data = pd.merge(train_data,bank_num,on=['id'],how='left')
    #%%
    train_data = pd.merge(train_data,train_order_time_max,on=['id'],how='left')
    train_data = pd.merge(train_data,train_order_time_min,on=['id'],how='left')
    train_data = pd.merge(train_data,train_order_type_zaixian,on=['id'],how='left')
    train_data = pd.merge(train_data,train_order_type_huodao,on=['id'],how='left')
    train_data = pd.merge(train_data,is_double_,on=['id'],how='left')
    train_data = pd.merge(train_data,is_0_0_0,on=['id'],how='left')
    train_data = pd.merge(train_data,is_1_1_1,on=['id'],how='left')
    train_data = pd.merge(train_data,is_0000_00_00,on=['id'],how='left')
    train_data = pd.merge(train_data,is_0001_1_1,on=['id'],how='left')
    train_data = pd.merge(train_data,is_hou_in,on=['id'],how='left')
    # train_data = pd.merge(train_data,is_nan,on=['id'],how='left')
    train_data = pd.merge(train_data,tmp_tmp_recieve,on=['id'],how='left')
    train_data = pd.merge(train_data,tmp_tmp_recieve_phone_count,on=['id'],how='left')
    train_data = pd.merge(train_data,tmp_tmp_recieve_phone_count_unique,on=['id'],how='left')
    train_data = pd.merge(train_data,bank_phone_num,on=['id'],how='left')
    train_data = pd.merge(train_data,is_hobby_df,on=['id'],how='left')
    train_data = pd.merge(train_data,is_idcard_df,on=['id'],how='left')
    train_data = pd.merge(train_data,auth_idcard_df,on=['id'],how='left')
    train_data = pd.merge(train_data,auth_phone_df,on=['id'],how='left')
    
#    "增加特征"
#    train_data = pd.merge(train_data,train_bankcard_bank_count,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_bankcard_card_count,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_bankcard_phone_count,on=['id'],how='left')
   
#    train_data = pd.merge(train_data,auth_time_df,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_order_mean_unit_price,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_order_mean_amt_order,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_order_phone_unique,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_order_many_success,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_order_many_occuer,on=['id'],how='left')
    #%%
    train_data['day_order_max'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['train_order_time_max']).days,axis=1);train_data = train_data.drop(['train_order_time_max'],axis=1)
    train_data['day_order_min'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['train_order_time_min']).days,axis=1);train_data = train_data.drop(['train_order_time_min'],axis=1)
    #%%order_info
    order_time = train_order.groupby(by = ['id'],as_index=False)['amt_order'].agg({'order_time':len})
    order_mean = train_order.groupby(by = ['id'],as_index=False)['amt_order'].agg({'order_mean':np.mean})
    unit_price_mean = train_order.groupby(by = ['id'],as_index=False)['unit_price'].agg({'unit_price_mean':np.mean})
    order_time_set = train_order.groupby(by = ['id'],as_index=False)['time_order'].agg({'order_time_set':lambda x:len(set(x))})
    
#   "4_19"
    # _loan = pd.merge(train_order[['time_order','amt_order','id']], train_target[['appl_sbm_tm','id']],on=['id'],how='right')
    # before_loan = _loan[_loan.time_order<=_loan.appl_sbm_tm]
    # after_loan = _loan[_loan.time_order>_loan.appl_sbm_tm]
    # before_loan_time = before_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'before_loan_time':len})
    # after_loan_time = after_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'after_loan_time':len})
    # before_loan_mean = before_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'before_loan_mean':np.mean})
    # after_loan_mean = after_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'after_loan_mean':np.mean})
    # train_data = pd.merge(train_data,before_loan_time,on=['id'],how='left')
    # train_data = pd.merge(train_data,after_loan_time,on=['id'],how='left')
    # train_data = pd.merge(train_data,before_loan_mean,on=['id'],how='left')
    # train_data = pd.merge(train_data,after_loan_mean,on=['id'],how='left')

    #%%
    train_data = pd.merge(train_data,order_time,on=['id'],how='left')
    train_data = pd.merge(train_data,order_mean,on=['id'],how='left')
    train_data = pd.merge(train_data,order_time_set,on=['id'],how='left')
    train_data = pd.merge(train_data,unit_price_mean,on=['id'],how='left')

    if IS_OFFLine == False:
        train_data = train_data.drop(['appl_sbm_tm','id','id_card_x','auth_time','phone','birthday','hobby','id_card_y'],axis=1)
    
    if IS_OFFLine == True:
        dummy_fea = ['sex', 'merriage', 'income', 'qq_bound', 'degree', 'wechat_bound','account_grade','industry']
        dummy_df = pd.get_dummies(train_data.loc[:,dummy_fea])
        train_data_copy = pd.concat([train_data,dummy_df],axis=1)
        train_data_copy = train_data_copy.fillna(0)
        vaild_train_data = train_data_copy.drop(dummy_fea,axis=1)
        valid_train_train = vaild_train_data[vaild_train_data.appl_sbm_tm < datetime.datetime(2017,4,1)]
        valid_train_test = vaild_train_data[vaild_train_data.appl_sbm_tm >= datetime.datetime(2017,4,1)]
        valid_train_train = valid_train_train.drop(['appl_sbm_tm','id','id_card_x','auth_time','phone','birthday','hobby','id_card_y'],axis=1)
        valid_train_test = valid_train_test.drop(['appl_sbm_tm','id','id_card_x','auth_time','phone','birthday','hobby','id_card_y'],axis=1)
        vaild_train_x = valid_train_train.drop(['target'],axis=1)
        vaild_test_x = valid_train_test.drop(['target'],axis=1)
        redict_result, modelee = xgb_feature(vaild_train_x,valid_train_train['target'].values,vaild_test_x,None)
        print('valid auc',roc_auc_score(valid_train_test['target'].values,redict_result))
        sys.exit(23)
 
    #%%
    test_auth = pd.read_csv('../AI_risk_test_V3.0/test_auth_info.csv',parse_dates = ['auth_time'])
#    auth_time = test_auth['auth_time'].map(lambda x:0 if str(x)=='nan' else 1)
#    auth_time_df = pd.DataFrame();auth_time_df['id'] = test_auth['id'];auth_time_df['auth_time_df'] = auth_time
    auth_idcard = test_auth['id_card'].map(lambda x:0 if str(x)=='nan' else 1)
    auth_idcard_df = pd.DataFrame();auth_idcard_df['id'] = test_auth['id'];auth_idcard_df['auth_idcard_df'] = auth_idcard
    auth_phone = test_auth['phone'].map(lambda x:0 if str(x)=='nan' else 1)
    auth_phone_df = pd.DataFrame();auth_phone_df['id'] = test_auth['id'];auth_idcard_df['auth_phone_df'] = auth_phone
    #%%
    test_auth['auth_time'].replace('0000-00-00','nan',inplace=True)
    test_auth['auth_time'] = pd.to_datetime(test_auth['auth_time'])
    #%%
    test_bankcard = pd.read_csv('../AI_risk_test_V3.0/test_bankcard_info.csv')
    "增加特征"
    test_bankcard_bank_count = test_bankcard.groupby(by=['id'], as_index=False)['bank_name'].agg({'bankcard_count':lambda x :len(x)})
    test_bankcard_card_count = test_bankcard.groupby(by=['id'], as_index=False)['card_type'].agg({'card_type_count':lambda x :len(set(x))})
    test_bankcard_phone_count = test_bankcard.groupby(by=['id'], as_index=False)['phone'].agg({'phone_count':lambda x :len(set(x))})
    #%%
    test_credit = pd.read_csv('../AI_risk_test_V3.0/test_credit_info.csv')
    #信用评分反序
    test_credit['credit_score_inverse'] = test_credit['credit_score'].map(lambda x :605-x)
    #额度-使用值
    test_credit['can_use'] = test_credit['quota'] - test_credit['overdraft']
    #%%
    test_order = pd.read_csv('../AI_risk_test_V3.0/test_order_info.csv',parse_dates = ['time_order'])
    test_order['amt_order'] = test_order['amt_order'].map(lambda x:np.nan if ((x == 'NA')| (x == 'null')) else float(x))
    test_order['time_order'] = test_order['time_order'].map(lambda x : pd.lib.NaT if (str(x) == '0' or x == 'NA' or x == 'nan')
                            else (datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S') if ':' in str(x) 
                            else (datetime.datetime.utcfromtimestamp(int(x[0:10])) + datetime.timedelta(hours = 8))))
    test_order_time_max = test_order.groupby(by=['id'], as_index=False)['time_order'].agg({'test_order_time_max':lambda x:max(x)})
    test_order_time_min = test_order.groupby(by=['id'], as_index=False)['time_order'].agg({'test_order_time_min':lambda x:min(x)})
    test_order_type_zaixian = test_order.groupby(by=['id']).apply(lambda x:x['type_pay'][(x['type_pay']=='在线支付').values].count()).reset_index(name = 'type_pay_zaixian')
    test_order_type_huodao = test_order.groupby(by=['id']).apply(lambda x:x['type_pay'][(x['type_pay']=='货到付款').values].count()).reset_index(name = 'type_pay_huodao')
#    test_order_mean_unit_price = test_order.groupby(by=['id']).apply(lambda x:np.mean(x['unit_price'])).reset_index(name = 'mean_unit_price')
#    test_order_mean_amt_order = test_order.groupby(by=['id']).apply(lambda x:np.mean(x['amt_order'])).reset_index(name = 'mean_amt_order')
#    test_order_phone_unique = test_order.groupby(by=['id']).apply(lambda x:x['phone'].nunique()).reset_index(name = '_order_phone_unique')
#    test_order_many_success = test_order.groupby(by=['id']).apply(lambda x:x['sts_order'][(x['sts_order']=='完成').values].count()).reset_index(name = '_order_many_success')
#    test_order_many_occuer = test_order.groupby(by=['id']).apply(lambda x:x['sts_order'].count()).reset_index(name = '_order_many_occuer')
    #%%
    test_recieve = pd.read_csv('../AI_risk_test_V3.0/test_recieve_addr_info.csv')
    test_recieve['region'] = test_recieve['region'].map(lambda x:str(x)[:2])
    tmp_tmp_recieve = pd.crosstab(test_recieve.id,test_recieve.region);tmp_tmp_recieve = tmp_tmp_recieve.reset_index()
    tmp_tmp_recieve_phone_count = test_recieve.groupby(by=['id']).apply(lambda x:x['fix_phone'].count());tmp_tmp_recieve_phone_count=tmp_tmp_recieve_phone_count.reset_index()
    tmp_tmp_recieve_phone_count_unique = test_recieve.groupby(by=['id']).apply(lambda x:x['fix_phone'].nunique());tmp_tmp_recieve_phone_count_unique=tmp_tmp_recieve_phone_count_unique.reset_index()

    test_target = pd.read_csv('../AI_risk_test_V3.0/test_list.csv',parse_dates = ['appl_sbm_tm'])
    test_user = pd.read_csv('../AI_risk_test_V3.0/test_user_info.csv',parse_dates = ['birthday'])
    is_hobby = test_user['hobby'].map(lambda x:0 if str(x)=='nan' else 1)
    is_hobby_df = pd.DataFrame();is_hobby_df['id'] = test_user['id'];is_hobby_df['is_hobby'] = is_hobby
    is_idcard = test_user['id_card'].map(lambda x:0 if str(x)=='nan' else 1)
    is_idcard_df = pd.DataFrame();is_idcard_df['id'] = test_user['id'];is_idcard_df['is_hobby'] = is_idcard
    #%%usesr_birthday
    tmp_tmp = train_user[['id','birthday']];tmp_tmp = tmp_tmp.set_index(['id'])
    is_double_ = tmp_tmp['birthday'].map(lambda x:(str(x) == '--')*1).reset_index(name='is_double_')
    is_0_0_0 = tmp_tmp['birthday'].map(lambda x:(str(x) == '0-0-0')*1).reset_index(name='is_0_0_0')
    is_1_1_1 = tmp_tmp['birthday'].map(lambda x:(str(x) == '1-1-1')*1).reset_index(name='is_1_1_1')
    is_0000_00_00 = tmp_tmp['birthday'].map(lambda x:(str(x) == '0000-00-00')*1).reset_index(name='is_0000_00_00')
    is_0001_1_1 = tmp_tmp['birthday'].map(lambda x:(str(x) == '0001-1-1')*1).reset_index(name='is_0001_1_1')
    is_hou_in = tmp_tmp['birthday'].map(lambda x:('后' in str(x))*1).reset_index(name='is_hou_in')
    # is_nan = tmp_tmp['birthday'].map(lambda x:(str(x) == 'nan')*1).reset_index(name='is_nan')
    #%%
    test_user['birthday'] = test_user['birthday'].map(lambda x:datetime.datetime.strptime(str(x),'%Y-%m-%d') if( re.match('19\d{2}-\d{1,2}-\d{1,2}',str(x)) and '-0' not in str(x)) else pd.lib.NaT)
    
    test_data = pd.merge(test_target,test_auth,on=['id'],how='left')
    test_data = pd.merge(test_data,test_user,on=['id'],how='left')
    test_data = pd.merge(test_data,test_credit,on=['id'],how='left')
    
    test_data['hour'] = test_data['appl_sbm_tm'].map(lambda x:x.hour)
    test_data['month'] = test_data['appl_sbm_tm'].map(lambda x:x.month)
    test_data['year'] = test_data['appl_sbm_tm'].map(lambda x:x.year)
    test_data['quota_use_ratio'] = test_data['overdraft'] / (test_data['quota']+0.01)
    test_data['nan_num'] = test_data.isnull().sum(axis=1)
    test_data['diff_day'] = test_data.apply(lambda row: (row['appl_sbm_tm'] - row['auth_time']).days,axis=1)
    test_data['how_old'] = test_data.apply(lambda row: (row['appl_sbm_tm'] - row['birthday']).days/365,axis=1)
    #################################################################
    auth_idcard = list(test_data['id_card_x']);user_idcard = list(test_data['id_card_y'])
    idcard_result = []
    for indexx,uu in enumerate(auth_idcard):
        if (str(auth_idcard[indexx])=='nan') and (str(user_idcard[indexx])=='nan'):
            idcard_result.append(0)
        elif (str(auth_idcard[indexx])!='nan') and (str(user_idcard[indexx])=='nan'):
            idcard_result.append(1)
        elif (str(auth_idcard[indexx])=='nan') and (str(user_idcard[indexx])!='nan'):
            idcard_result.append(2)
        else:
            ttt1 = str(auth_idcard[indexx])[0] + str(auth_idcard[indexx])[-1]
            ttt2 = str(user_idcard[indexx])[0] + str(user_idcard[indexx])[-1]
            if ttt1 == ttt2:
                idcard_result.append(3)
            if ttt1 != ttt2:
                idcard_result.append(4)
    test_data['the_same_id'] = idcard_result

    test_bankcard_phone_list = test_bankcard.groupby(by=['id'])['phone'].apply(lambda x:list(set(x.tolist()))).reset_index(name = 'bank_phone_list')
    test_data = pd.merge(test_data,test_bankcard_phone_list,on=['id'],how='left')
    test_data['exist_phone'] = test_data.apply(lambda x:x['phone'] in x['bank_phone_list'],axis=1)
    test_data['exist_phone'] = test_data['exist_phone']*1
    test_data = test_data.drop(['bank_phone_list'],axis=1)
    #%%bankcard_info
    bank_name = test_bankcard.groupby(by= ['id'], as_index= False)['bank_name'].agg({'bank_name_len':lambda x:len(set(x))})
    bank_num = test_bankcard.groupby(by= ['id'], as_index = False)['tail_num'].agg({'tail_num_len':lambda x:len(set(x))})
    bank_phone_num = test_bankcard.groupby(by= ['id'], as_index = False)['phone'].agg({'bank_phone_num':lambda x:x.nunique()})    
    
    
    test_data = pd.merge(test_data,bank_name,on=['id'],how='left')
    test_data = pd.merge(test_data,bank_num,on=['id'],how='left')
#%%
    test_data = pd.merge(test_data,test_order_time_max,on=['id'],how='left')
    test_data = pd.merge(test_data,test_order_time_min,on=['id'],how='left')
    test_data = pd.merge(test_data,test_order_type_zaixian,on=['id'],how='left')
    test_data = pd.merge(test_data,test_order_type_huodao,on=['id'],how='left')
    test_data = pd.merge(test_data,is_double_,on=['id'],how='left')
    test_data = pd.merge(test_data,is_0_0_0,on=['id'],how='left')
    test_data = pd.merge(test_data,is_1_1_1,on=['id'],how='left')
    test_data = pd.merge(test_data,is_0000_00_00,on=['id'],how='left')
    test_data = pd.merge(test_data,is_0001_1_1,on=['id'],how='left')
    test_data = pd.merge(test_data,is_hou_in,on=['id'],how='left')
    # test_data = pd.merge(test_data,is_nan,on=['id'],how='left')
    test_data = pd.merge(test_data,tmp_tmp_recieve,on=['id'],how='left')
    test_data = pd.merge(test_data,tmp_tmp_recieve_phone_count,on=['id'],how='left')
    test_data = pd.merge(test_data,tmp_tmp_recieve_phone_count_unique,on=['id'],how='left')
    test_data = pd.merge(test_data,bank_phone_num,on=['id'],how='left')
    test_data = pd.merge(test_data,is_hobby_df,on=['id'],how='left')
    test_data = pd.merge(test_data,is_idcard_df,on=['id'],how='left')
    test_data = pd.merge(test_data,auth_idcard_df,on=['id'],how='left')
    test_data = pd.merge(test_data,auth_phone_df,on=['id'],how='left')
    
#    "增加特征"
#    test_data = pd.merge(test_data,test_bankcard_bank_count,on=['id'],how='left')
#    test_data = pd.merge(test_data,test_bankcard_card_count,on=['id'],how='left')
#    test_data = pd.merge(test_data,test_bankcard_phone_count,on=['id'],how='left')
    
#    test_data = pd.merge(test_data,auth_time_df,on=['id'],how='left')
#    test_data = pd.merge(test_data,test_order_mean_unit_price,on=['id'],how='left')
#    test_data = pd.merge(test_data,test_order_mean_amt_order,on=['id'],how='left')
#    test_data = pd.merge(test_data,test_order_phone_unique,on=['id'],how='left')
#    test_data = pd.merge(test_data,test_order_many_success,on=['id'],how='left')
#    test_data = pd.merge(test_data,test_order_many_occuer,on=['id'],how='left')
    test_data['day_order_max'] = test_data.apply(lambda row: (row['appl_sbm_tm'] - row['test_order_time_max']).days,axis=1);test_data = test_data.drop(['test_order_time_max'],axis=1)
    test_data['day_order_min'] = test_data.apply(lambda row: (row['appl_sbm_tm'] - row['test_order_time_min']).days,axis=1);test_data = test_data.drop(['test_order_time_min'],axis=1)
    #%%order_info
    order_time = test_order.groupby(by = ['id'],as_index=False)['amt_order'].agg({'order_time':len})
    order_mean = test_order.groupby(by = ['id'],as_index=False)['amt_order'].agg({'order_mean':np.mean})
    unit_price_mean = test_order.groupby(by = ['id'],as_index=False)['unit_price'].agg({'unit_price_mean':np.mean})
    order_time_set = test_order.groupby(by = ['id'],as_index=False)['time_order'].agg({'order_time_set':lambda x:len(set(x))})

#   "4_19"
    # _loan = pd.merge(test_order[['time_order','amt_order','id']], test_target[['appl_sbm_tm','id']],on=['id'],how='right')
    # before_loan = _loan[_loan.time_order<=_loan.appl_sbm_tm]
    # after_loan = _loan[_loan.time_order>_loan.appl_sbm_tm]
    # before_loan_time = before_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'before_loan_time':len})
    # after_loan_time = after_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'after_loan_time':len})
    # before_loan_mean = before_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'before_loan_mean':np.mean})
    # after_loan_mean = after_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'after_loan_mean':np.mean})
    # test_data = pd.merge(test_data,before_loan_time,on=['id'],how='left')
    # test_data = pd.merge(test_data,after_loan_time,on=['id'],how='left')
    # test_data = pd.merge(test_data,before_loan_mean,on=['id'],how='left')
    # test_data = pd.merge(test_data,after_loan_mean,on=['id'],how='left')   
    
    test_data = pd.merge(test_data,order_time,on=['id'],how='left')
    test_data = pd.merge(test_data,order_mean,on=['id'],how='left')
    test_data = pd.merge(test_data,order_time_set,on=['id'],how='left')
    test_data = pd.merge(test_data,unit_price_mean,on=['id'],how='left')
    
    
    test_data = test_data.drop(['appl_sbm_tm','id','id_card_x','auth_time','phone','birthday','hobby','id_card_y'],axis=1)
    test_data['target'] = -1       
    
    
    # test_data.to_csv('8288test.csv',index=None)
    # train_data.to_csv('8288train.csv',index=None);sys.exit(32)
#%%
    dummy_fea = ['sex', 'merriage', 'income', 'qq_bound', 'degree', 'wechat_bound','account_grade','industry']



    train_test_data = pd.concat([train_data,test_data],axis=0,ignore_index = True)

    train_test_data = train_test_data.fillna(0)
    
    dummy_df = pd.get_dummies(train_test_data.loc[:,dummy_fea])
    
    
    train_test_data = pd.concat([train_test_data,dummy_df],axis=1)
    train_test_data = train_test_data.drop(dummy_fea,axis=1)
    
    train_train = train_test_data.iloc[:train_data.shape[0],:]
    test_test = train_test_data.iloc[train_data.shape[0]:,:]
    
    
    train_train_x = train_train.drop(['target'],axis=1)
    test_test_x = test_test.drop(['target'],axis=1)
    
    
    predict_result, modelee = xgb_feature(train_train_x,train_train['target'].values,test_test_x,None)
    ans = pd.read_csv('../AI_risk_test_V3.0/test_list.csv',parse_dates = ['appl_sbm_tm'])
    ans['PROB'] = predict_result
    ans = ans.drop(['appl_sbm_tm'],axis=1)
    minmin, maxmax = min(ans['PROB']),max(ans['PROB'])
    ans['PROB'] = ans['PROB'].map(lambda x:(x-minmin)/(maxmax-minmin))
    ans['PROB'] = ans['PROB'].map(lambda x:'%.4f' % x)
    ans.to_csv('../result/04094test.csv',index=None)
    
    
    
    
