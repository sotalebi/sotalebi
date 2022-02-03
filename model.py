import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from Cfgs import Cfgs


class ModelLoader(Cfgs):
    def __init__(self, loader):
        super(ModelLoader, self).__init__()

        self.kf = KFold(n_splits=self.FOLDS, shuffle=False)
        self.params = self.pars_RF if self.MODEL == 'RF' else self.pars_SVR
        self.pca = PCA(n_components=self.PCA_DIM, svd_solver='full', whiten=False)
        self.scaler = preprocessing.MinMaxScaler(feature_range=(self.SCALER_MIN, self.SCALER_MAX))
        self.normalizer = preprocessing.StandardScaler()
        self.SVR_set_num = self.count_subsets(self.SVR)
        self.RF_set_num = self.count_subsets(self.RF)
        self.y1_predict = None
        self.y2_predict = None
        self.loader = loader
        np.random.seed(self.SEED)

    @staticmethod
    def count_subsets(dic):
        return len(list(itertools.product(*list(dic.values()))))

    def fit(self, params):
        data = self.loader.data
        result = {'MODEL': self.MODEL}
        if self.MODEL == 'SVR':
            regr = svm.SVR(**params)
        elif self.MODEL == 'RF':
            regr = RandomForestRegressor(**params)
        else:
            raise ValueError('Wrong regression. please choose MODEL in Cfgs as SVR or RF')
        result['params'] = params
        estimator = self.define_estimator(regr)

        if not self.CROSS_VAL:
            estimator.fit(data['x1'], data['y1'])

            pred_train = estimator.predict(data['x1'])
            pred_test = estimator.predict(data['x2'])

            result['r2_train'] = r2_score(data['y1'], pred_train)
            result['rmse_train'] = np.sqrt(mean_squared_error(data['y1'], pred_train))
            result['rmse_percent_train'] = (result['rmse_train'] / np.mean(data['y1'])) * 100
            result['bias_train'] = (1 / len(pred_train)) * np.sum(data['y1'] - pred_train)



        else:
            rmse_train = []
            rmse_val = []
            rmse_percent_train = []
            rmse_percent_val = []
            r2_train = []
            r2_val = []
            bias_train = []
            bias_val = []

            kf = KFold(n_splits=self.FOLDS, shuffle=True)
            for train_index, val_index in kf.split(data['x1']):
                estimator.fit(data['x1'][train_index], data['y1'][train_index])

                pred_train = estimator.predict(data['x1'][train_index])
                pred_val = estimator.predict(data['x1'][val_index])

                rmse_value_train = np.sqrt(mean_squared_error(data['y1'][train_index], pred_train))
                rmse_value_val = np.sqrt(mean_squared_error(data['y1'][val_index], pred_val))

                rmse_train.append(rmse_value_train)
                rmse_val.append(rmse_value_val)

                rmse_percent_train.append((rmse_value_train / np.mean(data['y1'][train_index])) * 100)
                rmse_percent_val.append((rmse_value_val / np.mean(data['y1'][val_index])) * 100)

                r2_train.append(r2_score(data['y1'][train_index], pred_train))
                r2_val.append(r2_score(data['y1'][val_index], pred_val))

                bias_train.append((1 / len(pred_train)) * np.sum(data['y1'][train_index] - pred_train))
                bias_val.append((1 / len(pred_val)) * np.sum(data['y1'][val_index] - pred_val))

            result['r2_train'] = r2_train
            result['r2_val'] = r2_val

            result['rmse_train'] = rmse_train
            result['rmse_val'] = rmse_val

            result['rmse_percent_train'] = rmse_percent_train
            result['rmse_percent_val'] = rmse_percent_val

            result['bias_train'] = bias_train
            result['bias_val'] = bias_val

            # Final analysis on test set:
            estimator.fit(data['x1'], data['y1'])
            pred_test = estimator.predict(data['x2'])

        # Common computation whether cross val is used or not:
        result['r2_test'] = r2_score(data['y2'], pred_test)
        result['rmse_test'] = np.sqrt(mean_squared_error(data['y2'], pred_test))
        result['rmse_percent_test'] = (result['rmse_test'] / np.mean(data['y2'])) * 100
        result['bias_test'] = (1 / len(pred_test)) * np.sum(data['y2'] - pred_test)
        pred_test_dict = {'pred':pred_test, 'residual': data['y2']-pred_test}
        return result, pred_test_dict

    def define_estimator(self, regr):
        if self.PCA:
            estimator = make_pipeline(self.normalizer, self.pca, regr)
        else:
            estimator = make_pipeline(self.scaler, regr)
        return estimator

    def exhaustive_search(self):
        output = pd.DataFrame()
        for idx, item in tqdm(enumerate(itertools.product(*list(getattr(self, self.MODEL).values())))):
            pars = {}
            for index, key in enumerate(getattr(self, self.MODEL).keys()):
                pars[key] = item[index]
            result, _ = self.fit(pars)
            result.pop('MODEL')
            result.pop('params')
            starting_dict = {'MODEL': self.MODEL, **pars}
            df = {**starting_dict, **result}
            for k, v in df.items():
                if isinstance(v, list):
                    df[k] = np.mean(v)
            if idx == 0:
                output = pd.DataFrame(df, index=[0])
            else:
                output = output.append(df, ignore_index=True)
        if self.CROSS_VAL:
            output.sort_values(by='rmse_val', inplace=True, ascending=True)
        else:
            output.sort_values(by='rmse_test', inplace=True, ascending=True)
        output.to_csv('./Results/{}_{}.csv'.format(self.SAVE_NAME, self.MODEL), index=False)
