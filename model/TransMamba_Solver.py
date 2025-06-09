import csv

import torch
import time
from utils.utils import *
from model.TransMamba import TransMamba
from data_factory.data_loader import get_loader_segment
from metrics.metrics import *
import warnings
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from ptflops import get_model_complexity_info
TORCH_NN_DATA_PARALLEL = torch.nn.DataParallel
if torch.cuda.device_count() > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"cuda_num = {torch.cuda.device_count()}")

warnings.filterwarnings('ignore')

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    kl_loss = torch.mean(torch.sum(res, dim=-1), dim=1)
    return kl_loss

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class TransMamba_Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.config = config
        self.train_loader = get_loader_segment('dataset/' + self.config.dataset, batch_size=self.config.batch_size,
                                               win_size=self.config.win_size, mode='train',
                                               dataset=self.config.dataset)
        self.vali_loader = get_loader_segment('dataset/' + self.config.dataset, batch_size=self.config.batch_size,
                                              win_size=self.config.win_size, mode='val', dataset=self.config.dataset)
        self.test_loader = get_loader_segment('dataset/' + self.config.dataset, batch_size=self.config.batch_size,
                                              win_size=self.config.win_size, mode='test', dataset=self.config.dataset)
        self.thre_loader = get_loader_segment('dataset/' + self.config.dataset, batch_size=self.config.batch_size,
                                              win_size=self.config.win_size, mode='thre', dataset=self.config.dataset)

        self.build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.config.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = TransMamba(self.config)
        self.optimizer_T = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.optimizer_F = torch.optim.Adam(self.model.parameters(), lr=self.config.lr/2)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model.cuda()

        if self.config.show_params == 1:
            input_size = (self.config.win_size, self.config.channel)
            flops, params = get_model_complexity_info(self.model, input_size, as_strings=True, print_per_layer_stat=False)
            print('FLOPs: ', flops)
            print('Params: ', params)
            nParams = sum([p.nelement() for p in self.model.parameters()])
            print('Number of model parameters is', nParams)

    def vali(self, vali_loader):
        print('====================  Vali  ===================')
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(tqdm(vali_loader)):
            inputs = input_data.float().to(self.device)
            high_attention, low_attention, outputs, loss_cl, recon_loss_f = self.model(inputs)
            recon_loss_t = self.criterion(inputs, outputs)
            high_loss = 0.0
            low_loss = 0.0
            for u in range(len(low_attention)):
                # Normalize low attention
                norm_low_attention = low_attention[u] / torch.unsqueeze(torch.sum(low_attention[u], dim=-1),
                                                                        dim=-1).repeat(1, 1, 1, self.config.win_size)
                # High loss components
                high_loss += torch.mean(my_kl_loss(high_attention[u], norm_low_attention.detach())) + \
                             torch.mean(my_kl_loss(norm_low_attention.detach(), high_attention[u]))

                # Low loss components
                low_loss += torch.mean(my_kl_loss(norm_low_attention, high_attention[u].detach())) + \
                            torch.mean(my_kl_loss(high_attention[u].detach(), norm_low_attention))

            high_loss = high_loss / len(low_attention)
            low_loss = low_loss / len(low_attention)

            loss_1.append((low_loss - high_loss).item())
            loss_2.append((recon_loss_t + recon_loss_f + loss_cl).item())
        vali_loss1 = np.average(loss_1)
        vali_loss2 = np.average(loss_2)
        print(f"vali_loss1:{vali_loss1}, vali_loss2:{vali_loss2}")

        return vali_loss1, vali_loss2

    def train(self, setting):
        print('====================  Train  ===================')
        path = self.config.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(setting, patience=self.config.patience, verbose=True,
                                          dataset_name=self.config.dataset, delta=0.001)

        for epoch in range(self.config.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(tqdm(self.train_loader)):

                iter_count += 1
                inputs = input_data.float().to(self.device)
                high_attention, low_attention, outputs, loss_cl, recon_loss_f = self.model(inputs)
                recon_loss_t = self.criterion(inputs, outputs)

                high_loss = 0.0
                low_loss = 0.0

                for u in range(len(low_attention)):
                    # Normalize low attention
                    norm_low_attention = low_attention[u] / torch.unsqueeze(torch.sum(low_attention[u], dim=-1),
                                                                            dim=-1).repeat(1, 1, 1,
                                                                                           self.config.win_size)
                    # High loss components
                    high_loss += torch.mean(my_kl_loss(high_attention[u], norm_low_attention.detach())) + \
                                 torch.mean(my_kl_loss(norm_low_attention.detach(), high_attention[u]))

                    # Low loss components
                    low_loss += torch.mean(my_kl_loss(norm_low_attention, high_attention[u].detach())) + \
                                torch.mean(my_kl_loss(high_attention[u].detach(), norm_low_attention))

                high_loss = high_loss / len(low_attention)
                low_loss = low_loss / len(low_attention)

                loss_T = low_loss - high_loss
                loss_F = recon_loss_t + recon_loss_f + loss_cl

                # Step 1: Optimize loss_T with optimizer_T
                self.optimizer_T.zero_grad()
                loss_T.backward(retain_graph=True)
                self.optimizer_T.step()

                # Step 2: Optimize loss_F with optimizer_F
                self.optimizer_F.zero_grad()
                loss_F.backward()
                self.optimizer_F.step()
            print("Epoch: {0}, Cost time: {1:.3f}s ".
                  format(epoch + 1, time.time() - epoch_time))

            vali_loss1, vali_loss2 = self.vali(vali_loader=self.vali_loader)
            vali_loss = vali_loss1 + vali_loss2
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                break

            adjust_learning_rate(self.optimizer_T, epoch + 1, self.config.lr)
            adjust_learning_rate(self.optimizer_F, epoch + 1, self.config.lr/2)

    def test(self, setting=None):
        criterion = nn.MSELoss(reduce=False)
        with torch.no_grad():
            save_path = os.path.join(self.config.model_save_path, str(self.config.dataset))
            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module
            self.model.load_state_dict(
                torch.load(
                    os.path.join(save_path, str(self.config.dataset)  + setting + '_checkpoint.pth')))

            self.model.eval()
            temperature = 50
            # (1) stastic on the train set
            attens_energy = []
            for i, (input_data, labels) in enumerate(tqdm(self.train_loader)):
                inputs = input_data.float().to(self.device)
                high_attention, low_attention, outputs, loss_cl, recon_loss_f = self.model(inputs)
                recon_score = torch.softmax(torch.mean(criterion(inputs, outputs), dim=-1), dim=-1)
                high_loss = 0.0
                low_loss = 0.0
                for u in range(len(low_attention)):
                    # Normalize low attention
                    norm_low_attention = low_attention[u] / torch.unsqueeze(
                        torch.sum(low_attention[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.config.win_size)

                    # Compute KL losses with temperature
                    high_kl = my_kl_loss(high_attention[u], norm_low_attention) * temperature
                    low_kl = my_kl_loss(norm_low_attention, high_attention[u]) * temperature

                    # Accumulate losses
                    if u == 0:
                        high_loss, low_loss = high_kl, low_kl
                    else:
                        high_loss += high_kl
                        low_loss += low_kl

                sim_dis = torch.softmax((-high_loss - low_loss), dim=-1)
                sim_dis = sim_dis.detach().cpu().numpy()
                recon_score = recon_score.detach().cpu().numpy()

                score = sim_dis + recon_score

                attens_energy.append(score)
            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            train_energy = np.array(attens_energy)

            # (2) evaluation on the test set
            attens_energy = []
            test_labels = []
            for i, (input_data, labels) in enumerate(tqdm(self.thre_loader)):
                inputs = input_data.float().to(self.device)
                high_attention, low_attention, outputs, loss_cl, recon_loss_f = self.model(inputs)
                recon_score = torch.softmax(torch.mean(criterion(inputs, outputs), dim=-1), dim=-1)

                high_loss = 0.0
                low_loss = 0.0
                for u in range(len(low_attention)):
                    # Normalize low attention
                    norm_low_attention = low_attention[u] / torch.unsqueeze(
                        torch.sum(low_attention[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.config.win_size)

                    # Compute KL losses with temperature
                    high_kl = my_kl_loss(high_attention[u], norm_low_attention) * temperature
                    low_kl = my_kl_loss(norm_low_attention, high_attention[u]) * temperature

                    # Accumulate losses
                    if u == 0:
                        high_loss, low_loss = high_kl, low_kl
                    else:
                        high_loss += high_kl
                        low_loss += low_kl

                sim_dis = torch.softmax((-high_loss - low_loss), dim=-1)
                sim_dis = sim_dis.detach().cpu().numpy()
                recon_score = recon_score.detach().cpu().numpy()

                score = sim_dis + recon_score

                attens_energy.append(score)
                test_labels.append(labels)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            thresh = np.percentile(combined_energy, 100 - self.config.anomaly_ratio)
            print("Threshold :", thresh)

            # (3) find the threshold
            pred = (test_energy > thresh).astype(int)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)
            gt = test_labels.astype(int)

            # (4) detection adjustment
            gt, pred = adjustment(gt, pred)

            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
            print(
                "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".
                format(accuracy, precision, recall, f_score))

            return accuracy, precision, recall, f_score
