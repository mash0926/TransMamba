import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import sys
import pywt


TORCH_NN_DATA_PARALLEL = torch.nn.DataParallel

if torch.cuda.device_count() > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"cuda_num = {torch.cuda.device_count()}")

class EarlyStopping:
    def __init__(self, setting, patience=15, verbose=True, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.setting = setting

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):

        save_path = os.path.join(path, str(self.dataset))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if isinstance(model, TORCH_NN_DATA_PARALLEL):
            model = model.module
            torch.save(model.state_dict(),
                       os.path.join(save_path, str(self.dataset) + self.setting + '_checkpoint.pth'))
        else:
            torch.save(model.state_dict(),
                       os.path.join(save_path, str(self.dataset) + self.setting + '_checkpoint.pth'))

        self.val_loss_min = val_loss

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate if epoch < 20 else args.learning_rate * (0.5 ** ((epoch // 20) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * (0.5 ** ((epoch // 10) // 1))}
    elif args.lradj == 'type6':
        lr_adjust = {20: args.learning_rate * 0.5, 40: args.learning_rate * 0.01, 60: args.learning_rate * 0.01,
                     8: args.learning_rate * 0.01, 100: args.learning_rate * 0.01}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))

class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

def random_masking(x, rate):
    B, W, C = x.shape
    miss_size = int(rate * W)
    if miss_size == 0:
        return x
    x_mask = x.clone()
    mid_start = (W - miss_size) // 2
    mid_end = mid_start + miss_size
    end_start = W - miss_size
    x_mask[:, mid_start:mid_end, :] = 0
    x_mask[:, end_start:, :] = 0
    return x_mask

def point_noise(x, rate=0.1, mean=0.0, std=1):
    B, W, C = x.shape
    num_anomalies = int(W * rate)
    noise = torch.normal(mean=mean, std=std, size=(B, num_anomalies, C), device=x.device)
    anomaly_indices = torch.randint(0, W, (num_anomalies,))
    x_noise = x.clone()
    x_noise[:, anomaly_indices, :] += noise
    return x_noise


def seg_ano(x, rate):
    B, W, C = x.shape
    aug_size = int(rate * B)
    idx_1 = torch.randint(0, B, (aug_size,), device=x.device)
    idx_2 = torch.randint(0, B, (aug_size,), device=x.device)

    while torch.any(idx_1 == idx_2):
        idx_2 = torch.randint(0, B, (aug_size,), device=x.device)
    time_start = torch.randint(7, W, (aug_size,), device=x.device)
    for i in range(len(idx_1)):
        x[idx_1[i], :, time_start[i]:] = x[idx_2[i], :, time_start[i]:]
    return x


def fine_data_augmentation(x, masking_data_rate=0.05, point_ano_rate=0.05, seg_ano_rate=0.05):
    if masking_data_rate:
        x = random_masking(x, masking_data_rate)
    if point_ano_rate:
        x = point_noise(x, point_ano_rate)
    if seg_ano_rate:
        x = seg_ano(x, seg_ano_rate)
    return x


def coarse_data_augmentation(x, wavelet='db4', level=4, threshold_factor=0.12, choice='DWT', kernel_size=3, stride=1):
    B, W, C = x.shape
    denoised_x = torch.zeros_like(x)
    if choice == 'DWT':
        for b in range(B):
            for c in range(C):
                signal = x[b, :, c].cpu().numpy()

                coeffs = pywt.wavedec(signal, wavelet, level=level)

                threshold = threshold_factor * np.max(np.abs(coeffs[-1]))
                coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

                denoised_signal = pywt.waverec(coeffs_denoised, wavelet)
                denoised_x[b, :, c] = torch.from_numpy(denoised_signal[:W]).to('cuda')
    elif choice == 'MA':
        ma_filter = moving_avg(kernel_size=kernel_size, stride=stride).to(x.device)
        denoised_x = ma_filter(x)
    return denoised_x


def dual_augmentation(x, aug_rate=0.1):
    x_fine = fine_data_augmentation(x, point_ano_rate=aug_rate,
                                    masking_data_rate=aug_rate,
                                    seg_ano_rate=aug_rate)
    x_coarse = coarse_data_augmentation(x, wavelet='db1', threshold_factor = 0.12, choice='MA')
    return x_fine, x_coarse
