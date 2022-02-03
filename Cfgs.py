class Cfgs:
    def __init__(self):
        self.PATH = './newmethod1/'  # path to tif files
        self.INPUT = 'all'  # ['vod', 'cwd', 'evap', 'prep', 'temp', 'veg', 'evi', 'ndvi']
        self.SPLIT = 0.2
        self.FOLDS = 5  # used when self.CROSS_VAL = True
        self.PCA = False
        self.PCA_DIM = 250
        self.CROSS_VAL = False
        self.SCALER_MIN = -1
        self.SCALER_MAX = 1

        self.MODE = '2d'
        self.PTAD = False
        self.PERMUTE = False  # When PTAD is True this option should be false
        self.FEAT_PERMUTE = 'tempmean20101239'  # permute a feature by its name if PERMUTE is True
        self.MODEL = 'RF'
        self.pars_SVR = {'C': 100, 'kernel': 'rbf'}  # One model training
        self.pars_RF = {'n_estimators': 500,
                        'min_samples_leaf': 15,
                        'min_samples_split': 3,
                        'max_features': 'sqrt'}  # One model training
        self.SEED = 31
        self.VERTICAL = 603
        self.HORIZONTAL = 1526
        self.EXHAUSTIVE_SEARCH = False
        self.SVR = {'C': [1e-2, 1e-1, 1, 1e1, 1e2],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [1, 2, 3, 4, 5, 6],
                    'gamma': ['scale', 'auto']}  # Grid search hyper-parameter tuning
        self.RF = {'n_estimators': [300,500,1000],
                   'min_samples_split': [3, 5, 7],
                   'max_depth': [8, 10, 15],
                   'max_features': ["sqrt", "log2"]}  # Grid search hyper-parameter tuning
        self.SAVE_NAME = 'GridSearch'

        self.SAVE_PRED = True
