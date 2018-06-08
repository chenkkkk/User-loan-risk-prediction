# -*- coding: utf-8 -*-
from heamy.dataset import Dataset
from heamy.estimator import Regressor, Classifier
from heamy.pipeline import ModelsPipeline
import pandas as pd
import xgboost as xgb
import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np



def xgb_feature(X_train, y_train, X_test, y_test=None):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'rank:pairwise',
              'eval_metric' : 'auc',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'seed': 1111,
              'silent':1
              }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test)
    model = xgb.train(params, dtrain, num_boost_round=800)
    predict = model.predict(dvali)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)

def xgb_feature2(X_train, y_train, X_test, y_test=None):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'rank:pairwise',
              'eval_metric' : 'auc',
              'eta': 0.015,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'seed': 11,
              'silent':1
              }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test)
    model = xgb.train(params, dtrain, num_boost_round=1200)
    predict = model.predict(dvali)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)

def xgb_feature3(X_train, y_train, X_test, y_test=None):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'rank:pairwise',
              'eval_metric' : 'auc',
              'eta': 0.01,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'seed': 1,
              'silent':1
              }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test)
    model = xgb.train(params, dtrain, num_boost_round=2000)
    predict = model.predict(dvali)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)


def et_model(X_train, y_train, X_test, y_test=None):
    model = ExtraTreesClassifier(max_features = 'log2', n_estimators = 1000 , n_jobs = -1).fit(X_train,y_train)
    return model.predict_proba(X_test)[:,1]

def gbdt_model(X_train, y_train, X_test, y_test=None):
    model = GradientBoostingClassifier(learning_rate = 0.02, max_features = 0.7, n_estimators = 700 , max_depth = 5).fit(X_train,y_train)
    predict = model.predict_proba(X_test)[:,1]
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)

def logistic_model(X_train, y_train, X_test, y_test=None):
    model = LogisticRegression(penalty = 'l2').fit(X_train,y_train)
    return model.predict_proba(X_test)[:,1]

def lgb_feature(X_train, y_train, X_test, y_test=None):
    lgb_train = lgb.Dataset(X_train, y_train,categorical_feature={'sex', 'merriage', 'income', 'qq_bound', 'degree', 'wechat_bound','account_grade','industry'})
    lgb_test = lgb.Dataset(X_test,categorical_feature={'sex', 'merriage', 'income', 'qq_bound', 'degree', 'wechat_bound','account_grade','industry'})
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'num_leaves': 25,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf':5,
        'max_bin':200,
        'verbose': 0,
    }
    gbm = lgb.train(params,
    lgb_train,
    num_boost_round=2000)
    predict = gbm.predict(X_test)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)

VAILD = False
if __name__ == '__main__':
    if VAILD == True:
        train_data = pd.read_csv('8288train.csv',engine = 'python');train_data = train_data.fillna(0)
        test_data = pd.read_csv('8288test.csv',engine = 'python');test_data = test_data.fillna(0)
        dummy_fea = ['sex', 'merriage', 'income', 'qq_bound', 'degree', 'wechat_bound','account_grade','industry']
        dummy_df = pd.get_dummies(train_data.loc[:,dummy_fea])
        train_data_copy = pd.concat([train_data,dummy_df],axis=1)
        train_data_copy = train_data_copy.fillna(0)
        vaild_train_data = train_data_copy.drop(dummy_fea,axis=1)
        valid_train_train = vaild_train_data[(vaild_train_data.year <= 2017) & (vaild_train_data.month < 4)]
        valid_train_test = vaild_train_data[(vaild_train_data.year >= 2017) & (vaild_train_data.month >= 4)]
        vaild_train_x = valid_train_train.drop(['target'],axis=1)
        vaild_test_x = valid_train_test.drop(['target'],axis=1)
        redict_result = logistic_model(vaild_train_x,valid_train_train['target'].values,vaild_test_x,None)
        print('valid auc',roc_auc_score(valid_train_test['target'].values,redict_result))



    # dummy_fea = ['sex', 'merriage', 'income', 'qq_bound', 'degree', 'wechat_bound','account_grade','industry']
    # for _fea in dummy_fea:
    #     print(_fea)
    #     le = LabelEncoder()
    #     le.fit(train_data[_fea].tolist())
    #     train_data[_fea] = le.transform(train_data[_fea].tolist())
    # train_data_copy = train_data.copy()
    # vaild_train_data = train_data_copy
    # valid_train_train = vaild_train_data[(vaild_train_data.year <= 2017) & (vaild_train_data.month < 4)]
    # valid_train_test = vaild_train_data[(vaild_train_data.year >= 2017) & (vaild_train_data.month >= 4)]
    # vaild_train_x = valid_train_train.drop(['target'],axis=1)
    # vaild_test_x = valid_train_test.drop(['target'],axis=1)

    # redict_result = lgb_feature(vaild_train_x,valid_train_train['target'].values,vaild_test_x,None)
    # print('valid auc',roc_auc_score(valid_train_test['target'].values,redict_result))


    if VAILD == False:
        train_data = pd.read_csv('8288train.csv',engine = 'python');train_data = train_data.fillna(0)
        test_data = pd.read_csv('8288test.csv',engine = 'python');test_data = test_data.fillna(0)
        train_test_data = pd.concat([train_data,test_data],axis=0,ignore_index = True)
        train_test_data = train_test_data.fillna(0)
        train_data = train_test_data.iloc[:train_data.shape[0],:]
        test_data = train_test_data.iloc[train_data.shape[0]:,:]
        dummy_fea = ['sex', 'merriage', 'income', 'qq_bound', 'degree', 'wechat_bound','account_grade','industry']
        for _fea in dummy_fea:
            print(_fea)
            le = LabelEncoder()
            le.fit(train_data[_fea].tolist() + test_data[_fea].tolist())
            tmp = le.transform(train_data[_fea].tolist() + test_data[_fea].tolist())
            train_data[_fea] = tmp[:train_data.shape[0]]
            test_data[_fea] = tmp[train_data.shape[0]:]
        train_x = train_data.drop(['target'],axis=1)
        test_x = test_data.drop(['target'],axis=1)
        lgb_dataset = Dataset(train_x,train_data['target'],test_x,use_cache=False)
        ##############################
        train_data = pd.read_csv('8288train.csv',engine = 'python');train_data = train_data.fillna(0)
        test_data = pd.read_csv('8288test.csv',engine = 'python');test_data = test_data.fillna(0)
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
        xgb_dataset = Dataset(X_train=train_train_x,y_train=train_train['target'],X_test=test_test_x,y_test=None,use_cache=False)
        #heamy
        model_xgb = Regressor(dataset=xgb_dataset, estimator=xgb_feature,name='xgb',use_cache=False)
        model_xgb2 = Regressor(dataset=xgb_dataset, estimator=xgb_feature2,name='xgb2',use_cache=False)
        model_xgb3 = Regressor(dataset=xgb_dataset, estimator=xgb_feature3,name='xgb3',use_cache=False)
        model_lgb = Regressor(dataset=lgb_dataset, estimator=lgb_feature,name='lgb',use_cache=False)
        model_gbdt = Regressor(dataset=xgb_dataset, estimator=gbdt_model,name='gbdt',use_cache=False)
        pipeline = ModelsPipeline(model_xgb,model_xgb2,model_xgb3,model_lgb,model_gbdt)
        stack_ds = pipeline.stack(k=5, seed=111, add_diff=False, full_test=True)
        stacker = Regressor(dataset=stack_ds, estimator=LinearRegression,parameters={'fit_intercept': False})
        predict_result = stacker.predict()
        ans = pd.read_csv('../AI_risk_test_V3.0/test_list.csv',parse_dates = ['appl_sbm_tm'])
        ans['PROB'] = predict_result
        ans = ans.drop(['appl_sbm_tm'],axis=1)
        minmin, maxmax = min(ans['PROB']),max(ans['PROB'])
        ans['PROB'] = ans['PROB'].map(lambda x:(x-minmin)/(maxmax-minmin))
        ans['PROB'] = ans['PROB'].map(lambda x:'%.4f' % x)
        ans.to_csv('./ans_stacking.csv',index=None)




















