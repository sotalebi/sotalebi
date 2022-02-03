import csv
import os.path
import utility
import loading
import numpy as np
from model import ModelLoader

loader = loading.DataLoading()
loader.tif2mat()


def print_result(res, feat=None):
    if feat is not None:
        print('{:20s}'.format('Permute Feature'), ':\t', feat)
    for k, val in res.items():
        if k == 'params':
            for pk, pv in val.items():
                print(pk, ':\t', pv)
            print('=' * 20)
        elif k == 'MODEL':

            print(k, ':\t', val)
        else:
            if isinstance(val, float):
                print('{:20s}'.format(k), ':\t', '{:.4f}'.format(val))
            else:
                val = np.around(val, 3)
                print('{:20s}'.format(k), ':\t', val, '   mean------>',
                      '{:.4f}'.format(np.mean(val)), '\n')
    print('=' * 20)


if not loader.PTAD:
    loader.prepare3d()
    model_loader = ModelLoader(loader)
    if loader.EXHAUSTIVE_SEARCH:
        model_loader.exhaustive_search()
    else:
        result, prediction = model_loader.fit(params=model_loader.params)
        if loader.SAVE_PRED:
            utility.save_2d(row=loader.VERTICAL, col=loader.HORIZONTAL,
                            index=loader.valid_values_index, value=prediction['pred'], model=loader.MODEL,
                            name='pred_test.tif', path=loader.PATH)
            utility.save_2d(row=loader.VERTICAL, col=loader.HORIZONTAL,
                            index=loader.valid_values_index, value=prediction['residual'], model=loader.MODEL,
                            name='residual.tif', path=loader.PATH)
        print_result(result)

else:
    assert not loader.EXHAUSTIVE_SEARCH, " PTAD and Exhaustive search can not be both True"
    if not os.path.isfile('result_permutation.csv'):
        with open('result_permutation.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            if loader.CROSS_VAL:
                first_row = ['MODEL', 'Feature', 'r2_train', 'r2_val_mean', 'rmse_train', 'rmse_val_mean',
                             'rmse_percent_train', 'rmse_percent_val_mean', 'bias_train', 'bias_val_mean',
                             'r2_test', 'rmse_test', 'rmse_percent_test', 'bias_test']
            else:
                first_row = ['MODEL', 'Feature', 'r2_train', 'rmse_train', 'rmse_percent_train', 'bias_train',
                             'r2_test', 'rmse_test', 'rmse_percent_test', 'bias_test']
            writer.writerow(first_row)
        f.close()
    for name in loader.input_dict.keys():
        if name == 'bio':
            continue
        loader.FEAT_PERMUTE = name
        feat_mat = np.copy(getattr(loader, name))
        loader.prepare3d()
        model_loader = ModelLoader(loader)
        result, _ = model_loader.fit(params=model_loader.params)
        print_result(result, feat=name)
        setattr(loader, loader.FEAT_PERMUTE, feat_mat)
        result.pop('params')
        field = list(result.values())
        for i, v in enumerate(field):
            if isinstance(v, list):
                field[i] = np.mean(v).round(6)
            if isinstance(v, float):
                field[i] = np.round(v, 6)
        field.insert(1, name)
        with open('result_permutation.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(field)

# sb.pairplot(self.df, diag_kind='kde', plot_kws=dict(s=0.1, edgecolor=None))
