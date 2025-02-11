from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import  TimeCapsule
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, ema_update, mean_filter
from utils.metrics import metric
from thop import profile

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
# import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        _, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')
        B, T, V = next(iter(self.train_loader))[0].shape
        L = self.args.level_dim
        self.d_inputs = [T, L, V]
        self.d_folds = [V * L, self.args.d_compress[0] * V, self.args.d_compress[0] * self.args.d_compress[1]]
        self.model = self._build_model().to(self.device)

        input = torch.zeros(1, T, V).cuda()
        flops, params = profile(self.model, inputs=(input, ))
        print('Flops: % .4fG'%(flops / 1e9))
        print('params: % .4fM'% (params / 1e6)) 

    def _build_model(self):
        model_dict = {
            'TimeCapsule': TimeCapsule,   # ours
        }
        if 'Time' in self.args.model:
            model = model_dict[self.args.model].Model(self.args, d_inputs=self.d_inputs, d_folds=self.d_folds,
                                                      d_compress=self.args.d_compress).float()
        else:
            model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _update_model_ema(self):
        return ema_update(self.model.y_encoder, self.model.x_encoder)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        L = self.args.level_dim
        iter_count = 0
        time_now = time.time()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                if L > 0:
                    batch_x = batch_x.float().to(self.device)  # (B, T, V)
                    batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'Time' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Time' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if (i + 1) % 500 == 0:
                    speed = (time.time() - time_now) / iter_count
                    print('\tinference speed: {:.4f}s/iter'.format(speed))
                    iter_count = 0
                    time_now = time.time()

                f_dim = -1 if self.args.features == 'MS' else 0

                if L > 0:
                    pred = outputs[0][:, -self.args.pred_len:, f_dim:].detach().cpu()
                    true = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu()
                else:
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.mean(total_loss)
        # self.model.train()
        return total_loss

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion2 = nn.HuberLoss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        jepa_losses = []
        scheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=2, gamma=self.args.gamma)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            jepa_loss = []
            loss_j = 0
            L = self.args.level_dim 

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'Time' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'Time' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0

                    # we add the following to adapt for the needs of timecapsule model
                    if L > 0:
                        if self.args.jepa:
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            # The following may change the dynamics of backpropagation due to the dropout
                            capsule_y, _ = self.model.jepa_forward(batch_x.shape, batch_y) 
                            loss_j = criterion2(outputs[1], capsule_y) 
                            jepa_loss.append(loss_j.item())
                        outputs = outputs[0][:, -self.args.pred_len:, f_dim:]
                    else:
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    loss_o = criterion2(outputs, batch_y[:, -self.args.pred_len:, f_dim:])
                    if epoch <= 100:
                        loss = loss_o + loss_j
                    else:
                        loss = loss_o 


                    train_loss.append(loss_o.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_o.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    # EMA updates the enc_y
                    if self.args.jepa:
                        self.model.y_encoder = self._update_model_ema()

                # if self.args.lradj == 'TST':
                #     adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                #     scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.mean(train_loss)
            jepa_loss = np.mean(jepa_loss)
            jepa_losses.append(jepa_loss)   # record the variation of JEPA for plotting
            vali_loss = self.vali(self.vali_data, self.vali_loader, criterion2)  # use the same loss function as in training phase
            test_loss = self.vali(self.test_data, self.test_loader, criterion)   # use test indicator

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Jepa Loss: {3:.7f}"
                  " Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(epoch + 1, train_steps, train_loss, jepa_loss, vali_loss, test_loss))
            # save model in every early stage
            if epoch <= 0.2 * self.args.train_epochs:
                print(f'Validation loss ({vali_loss:.6f}).  Saving model ...')
                torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                # adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
                scheduler.step()
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        np.save(path + '/' + 'jepa', jepa_losses)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        L = self.args.level_dim
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                    
                batch_x = batch_x.float().to(self.device)  # (B, T, V, L)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' or 'Time' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                pred = outputs[0][:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                true = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()

                if i == 1:
                    store = outputs[-1]
                    Ys = outputs[-2]
                preds += list(pred.flatten())
                trues += list(true.flatten())

                # x_sta, x_mean, x_std = mean_filter(batch_x)
                # y_sta, y_mean, y_std = mean_filter(batch_y[:, -self.args.pred_len:, f_dim:], 5)
                # pre_sta, pre_mean, pre_std = mean_filter(outputs[0][:, -self.args.pred_len:, f_dim:], 5)

                # x_sta = x_sta.detach().cpu().numpy()
                # x_mean = x_mean.detach().cpu().numpy()
                # x_std = x_std.detach().cpu().numpy()

                # y_sta = y_sta.detach().cpu().numpy()
                # y_mean = y_mean.detach().cpu().numpy()
                # y_std = y_std.detach().cpu().numpy()

                # pre_sta = pre_sta.detach().cpu().numpy()
                # pre_mean = pre_mean.detach().cpu().numpy()
                # pre_std = pre_std.detach().cpu().numpy()

                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                #     # sta
                #     gt_sta = np.concatenate((x_sta[0, :, -1], y_sta[0, :, -1]), axis=0)
                #     pd_sta = np.concatenate((x_sta[0, :, -1], pre_sta[0, :, -1]), axis=0)
                #     visual(gt_sta, pd_sta, os.path.join(folder_path, str(i) + 'sta.pdf'))

                #     # mean
                #     gt_mean = np.concatenate((x_mean[0, :, -1], y_mean[0, :, -1]), axis=0)
                #     pd_mean = np.concatenate((x_mean[0, :, -1], pre_mean[0, :, -1]), axis=0)
                #     visual(gt_mean, pd_mean, os.path.join(folder_path, str(i) + 'mean.pdf'))

                #     # std
                #     gt_std = np.concatenate((x_std[0, :, -1], y_std[0, :, -1]), axis=0)
                #     pd_std = np.concatenate((x_std[0, :, -1], pre_std[0, :, -1]), axis=0)
                #     visual(gt_std, pd_std, os.path.join(folder_path, str(i) + 'std.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        preds = np.array(preds)
        trues = np.array(trues)
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # The following savings are for model analyses
        # torch.save(store, folder_path + 'store.pt')
        # torch.save(Ys, folder_path + 'Ys.pt')
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
