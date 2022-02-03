import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from osgeo import gdal
import cnn
import utility
from Cfgs import Cfgs
import pandas as pd
import os


class DataLoading(Cfgs):
    def __init__(self):
        super(DataLoading, self).__init__()
        self.path = self.PATH
        self.names = self.extract_names()
        self.input_dict = self.input_dictionary()
        self.x = None
        self.y = None
        self.x3d = None
        self.x3d_train = None
        self.y3d = None
        self.y3d_train = None
        self.mask2d = None
        self.mask3d = None
        self.df = None
        self.mask2d_train = None
        self.mask2d_test = None
        self.mat_impute = None
        self.bio_scaler = None
        self.valid_values_index = None
        self.data = {}
        np.random.seed(self.SEED)

    def extract_names(self):
        names = []
        for i, name in enumerate(os.listdir(self.PATH)):
            if name.endswith(".tif"):
                name = name.replace('.tif', '')
                names.append(name)
                setattr(self, name, None)
        return names

    def input_dictionary(self):
        dictionary = {}
        for i, n in enumerate(self.names):
            if n != 'bio':
                dictionary[n] = i
        return dictionary

    def tif2mat(self):
        self.bio = gdal.Open(self.PATH + 'bio.tif')
        self.bio = self.bio.GetRasterBand(1).ReadAsArray()
        self.bio = self.remove_extra_row(self.bio, self.VERTICAL, self.HORIZONTAL)

        for var in self.input_dict.keys():
            # Open tif file
            tif = gdal.Open(self.PATH + var + '.tif')
            tif = tif.GetRasterBand(1).ReadAsArray()
            tif = tif.astype(np.float32)
            tif = self.remove_extra_row(tif, self.VERTICAL, self.HORIZONTAL)  # remove extra row
            setattr(self, var, tif)
        self.missvalue2nan()

        if self.PERMUTE and self.FEAT_PERMUTE is not None:
            self.feat_permute()

    def prepare2d(self):
        df_train = {}
        df_test = {}
        if self.PTAD:
            self.feat_permute()
        for k in self.input_dict.keys():
            if k != 'bio':
                x1 = getattr(self, k).reshape(-1)[self.mask2d_train.reshape(-1).nonzero()]
                x2 = getattr(self, k).reshape(-1)[self.mask2d_test.reshape(-1).nonzero()]
                df_train[k] = x1
                df_test[k] = x2
                # setattr(self, k, None)

        self.data['x1'] = pd.DataFrame(df_train).values
        self.data['x2'] = pd.DataFrame(df_test).values
        y1 = self.bio.reshape(-1)[self.mask2d_train.reshape(-1) != 0]
        y2 = self.bio.reshape(-1)[self.mask2d_test.reshape(-1) != 0]
        self.data['y1'] = y1
        self.data['y2'] = y2

        if self.SAVE_PRED:
            self.valid_values_index = self.mask2d_test.reshape(-1).nonzero()[0]
            valid_values = self.bio.reshape(-1)[self.valid_values_index]
            utility.save_2d(row=self.VERTICAL, col=self.HORIZONTAL, index=self.valid_values_index,
                            value=1, model=self.MODEL, name='mask.tif', path=self.PATH)

            utility.save_2d(row=self.VERTICAL, col=self.HORIZONTAL, index=self.valid_values_index,
                            value=valid_values, model=self.MODEL, name='bio_test.tif', path=self.PATH)

    def feat_permute(self):
        mat = getattr(self, self.FEAT_PERMUTE)
        idx = np.asarray(np.where(~np.isnan(mat))).T
        values = mat[idx[:, 0], idx[:, 1]]
        p = np.random.permutation(len(values))
        values = values[p]
        mat[idx[:, 0], idx[:, 1]] = values
        setattr(self, self.FEAT_PERMUTE, mat)

    def prepare3d(self):
        X_mat, Y_mat = self.mat_data(inputs=self.INPUT)
        self.mask2d = self.make_mask(np.concatenate((X_mat, Y_mat), axis=0), mode='2d')
        self.mask3d = self.make_mask(X_mat.copy(), mode='3d')

        # Split
        idxs = np.asarray(np.where(self.mask2d == 1.)).T
        np.random.seed(self.SEED)
        test_idx = idxs[np.random.choice(idxs.shape[0], int(self.SPLIT * len(idxs)), replace=False)]
        row_mask = test_idx[:, 0]
        col_mask = test_idx[:, 1]
        self.mask2d_train = self.mask2d.copy()
        self.mask2d_train[row_mask, col_mask] = 0.
        self.mask2d_test = self.mask2d - self.mask2d_train
        if self.MODE == '2d':
            self.prepare2d()
        else:
            self.x3d, self.x3d_train, _ = self.scale3d(X_mat, self.SCALER_MIN, self.SCALER_MAX, self.mask2d_test)
            self.y3d, self.y3d_train, self.bio_scaler = self.scale3d(Y_mat, self.SCALER_MIN, self.SCALER_MAX,
                                                                     self.mask2d_test)

    def mat_data(self, inputs='all'):  # Create 3d file
        if inputs == 'all':
            X_mat = np.empty((len(self.input_dict), self.VERTICAL, self.HORIZONTAL))
            keys = tuple(self.input_dict.keys())
            for i in range(len(self.input_dict)):
                X_mat[i, :, :] = getattr(self, keys[i])
        elif isinstance(inputs, list) or isinstance(inputs, tuple):
            X_mat = np.empty((len(inputs), self.VERTICAL, self.HORIZONTAL))
            for i in range(len(inputs)):
                X_mat[i, :, :] = getattr(self, inputs[i])
        else:
            raise ValueError('inputs must be a tuple or list of inputs or "all" to include all inputs')

        Y_mat = np.empty((1, self.VERTICAL, self.HORIZONTAL))
        Y_mat[0, :, :] = self.bio
        return X_mat, Y_mat

    @staticmethod
    def remove_extra_row(variable, rows, cols):
        return variable[:rows, :cols]

    def missvalue2nan(self):
        for name in self.names:
            var = getattr(self, name)
            var[var < -99998] = np.nan
            setattr(self, name, var)
        self.bio[self.bio < -99998] = np.nan
        self.bio[self.bio == 0] = np.nan

    @staticmethod
    def make_mask(tensor, mode):
        assert mode == '2d' or mode == '3d'
        if mode == '2d':
            output = ~np.isnan(np.sum(tensor, axis=0))
            return output.astype(float)
        else:
            tensor[~np.isnan(tensor)] = 1
            tensor[np.isnan(tensor)] = 0
            return tensor
            # return tensor.astype(bool).astype(float)

    def data2df(self):
        dic = {}
        if self.INPUT == 'all':
            for k in self.input_dict.keys():
                if k != 'bio':
                    dic[k] = getattr(self, k).reshape(-1)
        else:
            for k in self.INPUT:
                dic[k] = getattr(self, k)
        dic['bio'] = self.bio.reshape(-1)

        return pd.DataFrame(dic)

    @staticmethod
    def scale3d(tensor, min_value, max_value, mask2d_test):
        scaler = preprocessing.MinMaxScaler(feature_range=(min_value, max_value))
        shpe = tensor.shape
        output = np.empty_like(tensor)
        output_train = np.empty_like(tensor)
        for i in range(shpe[0]):
            mat = tensor[i, :, :].copy().reshape(-1)
            mat_train = tensor[i, :, :].copy()
            mat_train[mask2d_test.astype(bool)] = np.nan
            mat_train = mat_train.reshape(-1, 1)

            nan_idx = np.isnan(mat)
            nan_idx_train = np.isnan(mat_train)

            scaler.fit(mat_train)
            not_nan_scaled = scaler.transform(mat[~nan_idx].reshape(-1, 1)).reshape(-1)
            not_nan_scaled_train = scaler.transform(mat_train[~nan_idx_train].reshape(-1, 1)).reshape(-1)

            mat_final = np.empty_like(mat)
            mat_final_train = np.empty_like(mat_train)

            mat_final[nan_idx] = 0.
            mat_final[~nan_idx] = not_nan_scaled
            mat_final = mat_final.reshape(shpe[1:])

            mat_final_train[nan_idx_train] = 0.
            mat_final_train[~nan_idx_train] = not_nan_scaled_train
            mat_final_train = mat_final_train.reshape(shpe[1:])

            output[i, :, :] = mat_final
            output_train[i, :, :] = mat_final_train
        return output, output_train, scaler
