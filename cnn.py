import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import utility
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sch = 2000
kernel = (3, 3)
H = [302, 24, 2, 1]  # first element in H is
kernel_mul = False
base_lr = 1e-1
min_scale = -1
Epoch = 100000
seed = 30
torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
np.random.RandomState(seed)
np.random.seed(seed)
random.seed(seed)


def kernel_to_pad(kernel_size):
    return int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2)


def ratio(ks, msk3d, features=H[0]):
    pad = kernel_to_pad(ks)
    weight = torch.ones((1, features, ks[0], ks[1]),
                        dtype=torch.float).to(device)
    rtio = F.conv2d(msk3d, weight, padding=pad, bias=None,
                    groups=1) / (features * ks[0] * ks[1])

    return rtio


def lr(ep):
    if ep < 1000:
        return 1
    elif 1000 < ep < 5000:
        return 2
    elif 5000 < ep < 10000:
        return 3
    elif 10000 < ep < 15000:
        return 2
    elif 15000 < ep < 20000:
        return 1
    else:
        return 1


class Net(nn.Module):
    def __init__(self, ks, Hidden):
        super(Net, self).__init__()
        self.H = Hidden
        for i in range(len(self.H) - 1):
            setattr(self, 'conv{:d}'.format(i + 1),
                    nn.Conv2d(in_channels=self.H[i], out_channels=self.H[i + 1],
                              kernel_size=ks, padding=kernel_to_pad(ks)))
        self.tanh = nn.Tanh()
        self.Lrelu = nn.LeakyReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x, msk):
        for i in range(len(self.H) - 1):
            layer = getattr(self, 'conv{}'.format(i + 1))
            x = layer(x)
            if i != len(self.H) - 2:  # last layer does not consist of a Lrelu
                x = self.Lrelu(x)

            if kernel_mul and i != len(self.H) - 2:
                x = torch.mul(x, msk.repeat(1, self.H[i + 1], 1, 1))
        # x = torch.mul(x, msk)
        return x


def r2_compute(y, pred, mask):
    return r2_score(y.reshape(-1)[mask.reshape(-1).bool()]
                    .to('cpu').numpy(), pred.reshape(-1)
                    [mask.reshape(-1).bool()]
                    .detach().to('cpu').numpy())


def rmse_compute(y, pred, mask, scaler):
    MSE = mse(scaler.inverse_transform(y.reshape(-1)[mask.reshape(-1).bool()]
                                       .to('cpu').numpy().reshape(-1, 1)), scaler.inverse_transform
              (pred.reshape(-1)[mask.reshape(-1).bool()].detach().to('cpu').numpy().reshape(-1, 1)))
    rmse = np.sqrt(MSE)

    rmse_persent = (rmse / np.mean(scaler.inverse_transform(y.reshape(-1)[mask.reshape(-1).bool()]
                                   .to('cpu').numpy().reshape(-1, 1)))) * 100
    bias = (1 / len(pred.reshape(-1)[mask.reshape(-1).bool()].detach().to('cpu').numpy())) * \
           np.sum(y.reshape(-1)[mask.reshape(-1).bool()]
                  .to('cpu').numpy() - pred.reshape(-1)[mask.reshape(-1).bool()]
                  .detach().to('cpu').numpy())
    return rmse, rmse_persent, bias


def save_3d(r, c, value, mask, model, name, path):
    v = value.reshape(-1)[mask.detach().to('cpu').reshape(-1).bool()]
    idx = mask.reshape(-1).to('cpu').numpy()
    idx = list(idx.nonzero())[0]
    utility.save_2d(r, c, idx, v, model, name, path)


def unscale(array, mask, scaler):
    mat = array.squeeze().to('cpu').detach().numpy()
    mask = mask.squeeze().to('cpu').detach().numpy()
    mat[mask == 0] = np.nan
    idx = np.asarray(np.where(~np.isnan(mat))).T
    values = scaler.inverse_transform(mat[idx[:, 0], idx[:, 1]].reshape(-1, 1)).reshape(-1)
    mat[idx[:, 0], idx[:, 1]] = values
    return mat


def train(loader):
    mask3d = torch.from_numpy(loader.mask3d).unsqueeze(0).float().to(device)
    x = torch.from_numpy(loader.x3d).unsqueeze(0).float().to(device)
    y = torch.from_numpy(loader.y3d).unsqueeze(0).float().to(device)
    loader.mask2d_train = torch.from_numpy(loader.mask2d_train).unsqueeze(0).unsqueeze(0).float().to(device)
    loader.mask2d_test = torch.from_numpy(loader.mask2d_test).unsqueeze(0).unsqueeze(0).float().to(device)

    model = Net(ks=kernel, Hidden=H)
    model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=base_lr)
    kernel_ratio = ratio(ks=kernel, msk3d=mask3d)

    lambda1 = lambda ep: lr(ep)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # Create model:
    train_loss = []
    test_loss = []
    train_r2 = []
    test_r2 = []
    Y_train = torch.mul(y, loader.mask2d_train).to(device)
    Y_test = torch.mul(y, loader.mask2d_test).to(device)
    Y_pred_test = None

    model.train()
    for epoch in range(Epoch):

        Y_pred = model(x, kernel_ratio)

        Y_pred = torch.mul(Y_pred, loader.mask2d_train)

        model.eval()
        Y_pred_test = model(x, kernel_ratio)
        Y_pred_test = torch.mul(Y_pred_test, loader.mask2d_test)

        model.train()
        loss = criterion(Y_train, Y_pred)
        loss_test = criterion(Y_test, Y_pred_test)

        train_loss.append(loss.item())
        test_loss.append(loss_test.item())

        r2_train = r2_compute(Y_train, Y_pred, loader.mask2d_train)
        r2_test = r2_compute(Y_test, Y_pred_test, loader.mask2d_test)

        rmse_train, rmse_train_percent, bias_train = \
            rmse_compute(Y_train, Y_pred, loader.mask2d_train, loader.bio_scaler)
        rmse_test, rmse_test_percent, bias_test = \
            rmse_compute(Y_test, Y_pred_test, loader.mask2d_test, loader.bio_scaler)

        train_r2.append(r2_train)
        test_r2.append(r2_test)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 100 == 0:
            print('Epoch: {:<9d},Loss = {:<10.7f},\tTrain R2: {:<9.4f},\tTrain RMSE: {:<9.4f},'
                  '\tTest R2: {:<9.4f},\tTest RMSE: {:<9.4f},\tTest RMSE%: {:<9.4f},\tTest Bias: {:<9.4f},'
                  .format(epoch, loss.item(), r2_train, rmse_train, r2_test, rmse_test, rmse_test_percent, bias_test))

    if loader.SAVE_PRED:
        y_pred_test = unscale(Y_pred_test, loader.mask2d_test, loader.bio_scaler)
        y_test = unscale(Y_test, loader.mask2d_test, loader.bio_scaler)
        save_3d(r=loader.VERTICAL, c=loader.HORIZONTAL, value=y_pred_test, mask=loader.mask2d_test,
                model='cnn', name='cnn.tif', path=loader.PATH)
        save_3d(r=loader.VERTICAL, c=loader.HORIZONTAL, value=y_test - y_pred_test, mask=loader.mask2d_test,
                model='cnn', name='residual.tif', path=loader.PATH)
