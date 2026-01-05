import torch
import numpy as np
import torch.nn as nn

from copy import copy, deepcopy
import argparse
import os
import sys

sys.path.append("../utility/")
from torch.utils.tensorboard import SummaryWriter
import time

import Transformer_train

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.append("../RevNet/")
from iRevNet import iRevNet

from tqdm import tqdm
import gc

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class Network(nn.Module):
    def __init__(self, X_EMB_SIZE, U_EMB_SIZE, u_dim, x_dim):
        super(Network, self).__init__()

        '''
            Transformer
        '''
        self.SRC_SIZE = 3
        self.TGT_IZE = self.SRC_SIZE
        self.X_EMB_SIZE = X_EMB_SIZE
        self.NHEAD = 4
        self.FFN_HID_DIM = 128
        self.BATCH_SIZE = 128
        self.NUM_ENCODER_LAYERS = 3
        self.NUM_DECODER_LAYERS = 3
        self.MEMORY_STEP = 8

        self.SST_embedding_net = Transformer_train.SelfSupervisedTransformer(self.NUM_ENCODER_LAYERS,
                                                                             self.NUM_DECODER_LAYERS, self.X_EMB_SIZE,
                                                                             self.NHEAD, self.SRC_SIZE, self.TGT_IZE,
                                                                             self.BATCH_SIZE, self.MEMORY_STEP,
                                                                             self.FFN_HID_DIM)

        '''
            RevNet
        '''
        # 参数设置
        self.c_i = [4, 4, 4, 4]
        self.h_i = [64, 64, 128, 128]
        self.dropout_rate = 0.1
        self.affineBN = True
        self.in_shape = [u_dim]
        self.U_EMB_SIZE = U_EMB_SIZE

        self.U_RevNet = iRevNet(c_i=self.c_i, h_i=self.h_i, dropout_rate=self.dropout_rate, affineBN=self.affineBN,
                                in_shape=self.in_shape)

        self.u_dim = u_dim
        self.x_dim = x_dim
        self.lA = nn.Linear(self.X_EMB_SIZE + self.x_dim, self.X_EMB_SIZE + self.x_dim, bias=False)

        '''
            Quasi-diagonal Koopman
        '''
        self.h = np.mod(self.X_EMB_SIZE + self.x_dim, 2)
        self.g = (self.X_EMB_SIZE + self.x_dim - self.h) / 2
        with torch.no_grad():
            weight_matrix = torch.zeros(19, 19)
            for i in range(9):
                weight_matrix[2 * i:2 * i + 2, 2 * i:2 * i + 2] = gaussian_init_(2, std=1)
                weight_matrix[2 * i + 1, 2 * i] = - weight_matrix[2 * i, 2 * i + 1]
                weight_matrix[2 * i + 1, 2 * i + 1] = weight_matrix[2 * i, 2 * i]
            weight_matrix[-1, -1] = gaussian_init_(int(self.h), std=1)
            self.lA.weight.copy_(weight_matrix)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9

        self.lB = nn.Linear(self.U_EMB_SIZE, self.X_EMB_SIZE + self.x_dim, bias=False)

    def forward(self, x, b):
        return self.lA(x) + self.lB(b)

# loss function
def Klinear_loss(embedded_x, embedded_u, x_state, embedded_last_epoch_x, net, tgt_copy_net, mse_loss, K_shift, u_dim=1,
                 first_epoch=True, gamma=0.99):

    x_data = embedded_x.to(DEVICE)
    u_data = embedded_u.to(DEVICE)
    x_state = x_state.to(DEVICE)

    beta = 1.0
    alpha = 10e-7

    beta_sum = 0.0

    loss_linear = torch.zeros(1, dtype=torch.float64).to(DEVICE)
    loss_inf = torch.zeros(1, dtype=torch.float64).to(DEVICE)

    for i in range(0, K_shift):
        s_k_hat_i = x_data[i:-K_shift + i, :, :]
        u_k_hat_i = u_data[i:-K_shift + i, :, :]

        # beta_sum = beta_sum + beta

        if i < K_shift - 1:
            x_k_i = x_state[:, i+1:-K_shift + i + 1, :]
        else:
            x_k_i = x_state[:, i+1:, :]


        phi_k = torch.cat((x_state[:, i:-K_shift + i, :].permute(1, 0, 2), s_k_hat_i), dim=-1).detach() # 增广后状态，原状态+编码结果作为sk
        first = net.lA(phi_k)
        second = net.lB(u_k_hat_i.detach())
        for j in range(1, i + 1):

            second = second + net.lA(net.lB(u_data[i-j:-K_shift + i-j, :, :].detach()))
            first = net.lA(first)

        phi_hat_k_i = first + second

        s_k_i = tgt_copy_net.encoder(x_k_i, embedded_last_epoch_x, first_epoch)
        phi_k_i = torch.cat((x_k_i.permute(1, 0, 2), s_k_i), dim=-1)

        linear_loss = mse_loss(phi_hat_k_i, phi_k_i)

        Linf_den = torch.tensor(1.0, dtype=torch.float64)
        inf_loss = torch.norm(torch.norm(phi_k_i - phi_hat_k_i, dim=1, p=float('inf')), p=float('inf')) / Linf_den

        loss_linear = loss_linear + linear_loss
        loss_inf = loss_inf + alpha * inf_loss

        # beta *= gamma

    if first_epoch:
        loss_supervised = torch.zeros(1, dtype=torch.float64).to(DEVICE)
    else:
        loss_supervised = torch.zeros(1, dtype=torch.float64).to(DEVICE)

    loss = loss_linear / K_shift + loss_supervised * 0.3 + loss_inf

    return loss

def Eig_loss(net):
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs() - torch.ones(1, dtype=torch.float64).to(DEVICE)
    mask = c > 0
    loss = c[mask].sum()
    return loss

def update_state_dict(model, state_dict, tau: float, strip_ddp=True):
    if tau == 1.0:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {k: tau * state_dict[k] + (1 - tau) * v
                     for k, v in model.state_dict().items()}
        model.load_state_dict(update_sd)

def regulation(data):

    data = torch.tensor(data, requires_grad=False)

    mean = torch.mean(data)
    std = torch.std(data)

    data_normalized = (data - mean) / std

    data_normalized = (data_normalized - data_normalized.min()) / (data_normalized.max() - data_normalized.min())

    return data_normalized


class MyDataset2(Dataset):
    def __init__(self, data_tensor1):
        self.data_tensor1 = data_tensor1

    def __len__(self):
        return self.data_tensor1.shape[1]

    def __getitem__(self, idx):
        input_sequence = self.data_tensor1[:, idx, :]

        return input_sequence

def check(dynamic_type, Ktest_samples, Ktrain_samples, Ksteps, is_combine: bool):
    # 构建测试集
    if not os.path.exists("../Data/" + "TestData"):
        os.makedirs("../Data/" + "TestData")

    if is_combine:
        test_data_list = []
        for type in dynamic_type:
            test_data_list.append(np.load('../Data/TestData/CarModel_{}_Data_{}.npy'.format(type, Ktest_samples)))
        Ktest_data = np.concatenate(test_data_list, axis=1)
    else:
        Ktest_data = np.load('../Data/TestData/CarModel_{}_Data_{}.npy'.format(dynamic_type, Ktest_samples))

    # 构建训练集
    if not os.path.exists("../Data/" + "TrainData"):
        os.makedirs("../Data/" + "TrainData")

    if is_combine:
        train_data_list = []
        for type in dynamic_type:
            train_data_list.append(np.load('../Data/TrainData/CarModel_{}_Data_{}.npy'.format(type, Ktrain_samples)))
        Ktrain_data = np.concatenate(train_data_list, axis=1)
    else:
        Ktrain_data = np.load('../Data/TrainData/CarModel_{}_Data_{}.npy'.format(dynamic_type, Ktrain_samples))

    return Ktest_data, Ktrain_data

# main
def SST_Rev_Koopman(env_name, dynamic_type, is_combine, K_shift, epochs, suffix="", gamma=0.5,
                                                        Ktrain_samples=10000, Ktest_samples=2000):

    u_dim = 2

    if env_name == "CarModel":
        Ksteps = 10
        if is_combine:
            dynamic_type = ['Random', 'Skidpad', 'Fishhook', 'Slalom']
            Ktest_data, Ktrain_data = check(dynamic_type, Ktest_samples, Ktrain_samples, Ksteps, is_combine)
        elif dynamic_type == "Random":
            Ktest_data, Ktrain_data = check(dynamic_type, Ktest_samples, Ktrain_samples, Ksteps, is_combine)
        elif dynamic_type == "Skidpad":
            Ktest_data, Ktrain_data = check(dynamic_type, Ktest_samples, Ktrain_samples, Ksteps, is_combine)
        elif dynamic_type == "Fishhook":
            Ktest_data, Ktrain_data = check(dynamic_type, Ktest_samples, Ktrain_samples, Ksteps, is_combine)
        elif dynamic_type == "Slalom":
            Ktest_data, Ktrain_data = check(dynamic_type, Ktest_samples, Ktrain_samples, Ksteps, is_combine)

    Ktest_data = regulation(Ktest_data)
    Ktrain_data = regulation(Ktrain_data)

    Ktrain_samples = Ktrain_data.shape[1]
    x_dim = Ktest_data.shape[-1] - u_dim

    '''
        DeepKoopman部分
    '''
    torch.manual_seed(0)
    X_EMB_SIZE = 16
    U_EMB_SIZE = 8
    BATCH_SIZE = 128

    momentum_tau = 0.9

    net = Network(X_EMB_SIZE, U_EMB_SIZE, u_dim, x_dim)
    print("State Embedding Dimension:", X_EMB_SIZE)
    print("Input Embedding Dimension:", U_EMB_SIZE)

    learning_rate = 1e-4
    if torch.cuda.is_available():
        net.cuda()
    net.double()
    mse_loss = nn.MSELoss()
    parameters = list(net.SST_embedding_net.parameters()) + list(net.U_RevNet.parameters()) + list(net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # train
    eval_step = 5
    best_loss = 10.0

    writer = SummaryWriter(log_dir='../Data/Writer/')
    logdir = "../Data/"+suffix+"/SST_Rev_DeepKoopman_"+env_name+"_Xembedding{}_uembdding{}_samples{}".format(X_EMB_SIZE, U_EMB_SIZE, Ktrain_samples)
    if not os.path.exists( "../Data/"+suffix):
        os.makedirs( "../Data/"+suffix)


    X_train_iter = Ktrain_data[:, :, 2:]
    U_train_iter = Ktrain_data[:, :, :2]
    X_test_iter = Ktest_data[:, :, 2:]
    U_test_iter = Ktest_data[:, :, :2]

    x_dataset = MyDataset2(X_train_iter)
    u_dataset = MyDataset2(U_train_iter)
    x_test_dataset = MyDataset2(X_test_iter)
    u_test_dataset = MyDataset2(U_test_iter)

    X_train_data_loader = DataLoader(x_dataset, batch_size=BATCH_SIZE, shuffle=True)
    U_train_data_loader = DataLoader(u_dataset, batch_size=BATCH_SIZE, shuffle=True)
    X_test_data_loader = DataLoader(x_test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    U_test_data_loader = DataLoader(u_test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    '''
        Begin
    '''
    embedded_last_epoch_x = torch.zeros(Ktrain_data.shape[0], Ktrain_data.shape[1], net.X_EMB_SIZE)  # 保存上一轮的训练的embedding结果
    embedded_last_epoch_x_test_data = torch.zeros(Ktest_data.shape[0], Ktest_data.shape[1], net.X_EMB_SIZE)  # 保存上一轮的训练的embedding结果

    first_epoch = True
    first_epoch_test = True

    for i in range(1, epochs + 1):
        epoch_start_time = time.time()

        net.SST_embedding_net.train()
        net.U_RevNet.train()
        net.train()

        tgt_copy_net = deepcopy(net.SST_embedding_net)
        for param in (list(tgt_copy_net.parameters())):
            param.requires_grad = False

        iter = 0

        train_loss = 5

        pbar = tqdm(total=len(X_train_data_loader))

        if first_epoch:
            print("\nPre Processing, epoch num: {}".format(epochs))
        else:
            print("\nTraining Process begin")

        for x_state, u in zip(X_train_data_loader, U_train_data_loader):

            embedded_x = net.SST_embedding_net.encoder(x_state, embedded_last_epoch_x[:, BATCH_SIZE*iter:BATCH_SIZE*(iter + 1), :],
                                                       first_epoch)
            embedded_x = embedded_x.to(torch.float64)

            embedded_u = net.U_RevNet.encoder(u)
            embedded_u = embedded_u.to(torch.float64)

            if not first_epoch:
                loss = Klinear_loss(embedded_x, embedded_u, x_state,
                                    embedded_last_epoch_x[:, BATCH_SIZE*iter:BATCH_SIZE*(iter + 1), :],
                                    net, tgt_copy_net, mse_loss, K_shift, u_dim, first_epoch, gamma)

                update_state_dict(tgt_copy_net, net.SST_embedding_net.state_dict(), momentum_tau)      # EMA复制网络参数更新

                optimizer.zero_grad()
                if x_state.shape[0] < BATCH_SIZE:
                    if iter == 0:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward(retain_graph=False)
                else:
                    if iter == 0:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward(retain_graph=False)

                nn.utils.clip_grad_norm_(parameters, 10)
                optimizer.step()

                train_loss = train_loss + loss.item()

                embedded_last_epoch_x[:, BATCH_SIZE*iter:BATCH_SIZE*(iter + 1), :] = embedded_x     # 保存上一轮的embedding结果

                iter = iter + 1

                pbar.set_description(f'Epoch: {i}, Batch Loss: {np.round(loss.detach().cpu().numpy(), 4)}')
                print("\nEpoch: {}, Batch Loss: {}".format(i, np.round(loss.detach().cpu().numpy(), 4)))
                pbar.update(1)
                gc.collect()

                writer.add_scalar('Epoch/loss', loss, iter + (i-2)*(len(X_train_data_loader)))

            else:
                pbar.set_description('Processing the first epoch, please waiting...')
                pbar.update(1)

        train_loss = train_loss / len(X_train_data_loader)

        writer.add_scalar('Train/loss', train_loss, i)

        if first_epoch:
            first_epoch = False
        epoch_end_time = time.time()
        print("\nEpoch: {}, Train_loss_avg: {}, Time Consuming: {}min".format(i, np.round(train_loss, 4),
                                                                              round((epoch_end_time - epoch_start_time)/60, 2)))
        gc.collect()

        if i % eval_step == 0:
            # K loss
            print("\nTake a break! Testing process is running...")
            with torch.no_grad():

                net.SST_embedding_net.eval()
                net.U_RevNet.eval()
                net.eval()

                test_loss = 0

                iter = 0

                for x_state, u in zip(X_test_data_loader, U_test_data_loader):

                    # SST-Embedding
                    embedded_x = net.SST_embedding_net.encoder(x_state,
                                                               embedded_last_epoch_x_test_data[:, BATCH_SIZE*iter:BATCH_SIZE*(iter + 1), :],
                                                               first_epoch_test)
                    embedded_x = embedded_x.to(torch.float64)

                    embedded_u = net.U_RevNet.encoder(u)
                    embedded_u = embedded_u.to(torch.float64)

                    loss = Klinear_loss(embedded_x, embedded_u, x_state,
                                        embedded_last_epoch_x_test_data[:, BATCH_SIZE * iter:BATCH_SIZE * (iter + 1), :],
                                        net, tgt_copy_net, mse_loss, u_dim, K_shift, first_epoch_test, gamma)

                    test_loss = test_loss + loss

                    embedded_last_epoch_x_test_data[:, BATCH_SIZE * iter:BATCH_SIZE * (iter + 1), :] = embedded_x

                    iter = iter + 1
                    print("Iter: {}".format(iter))

                test_loss = test_loss / len(X_test_data_loader)
                test_loss = test_loss.detach().cpu().numpy()

                writer.add_scalar('Eval/best_loss', best_loss, i)
                writer.add_scalar('Eval/test_loss', test_loss, i)
                if test_loss < best_loss:
                    best_loss = copy(test_loss)
                    checkpoint = {
                                      'SST': net.SST_embedding_net,
                                      'RevNet': net.U_RevNet,
                                      'A': net.lA,
                                      'B': net.lB,
                                      'optimizer': optimizer
                                  }
                    torch.save(checkpoint, logdir + "_epoch{}".format(i) + ".pth")
                    print('Saved models with loss: {}'.format(best_loss))
                print("Step:{} Eval-loss{}".format(i, test_loss))
                # print("-------------END-------------")
                gc.collect()

        if first_epoch_test:
            first_epoch_test = False
        writer.add_scalar('Eval/best_loss', best_loss, i)
    print("END-best_loss{}".format(best_loss))

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.SST_embedding_net.encode(src, src_mask)

    ys = torch.zeros(1, 1, model.X_EMB_SIZE)
    tgt_mask = (model.SST_embedding_net.generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
    out = model.SST_embedding_net.decode(ys, memory, tgt_mask)

    return out


def main():
    if args.load_mode:
        SST_Rev_Koopman(args.env, args.dynamic_type, args.is_combine, args.shift, args.epoch, suffix=args.suffix,
                         gamma=args.gamma, Ktrain_samples=args.K_train_samples, Ktest_samples=args.K_test_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="CarModel")
    parser.add_argument("--dynamic_type", type=str, default="Skidpad")
    parser.add_argument("--is_combine", type=bool, default=True)

    parser.add_argument("--suffix", type=str, default="Weights")
    parser.add_argument("--K_train_samples", type=int, default=10000)
    parser.add_argument("--K_test_samples", type=int, default=2000)
    parser.add_argument("--augsuffix", type=str, default="")
    parser.add_argument("--all_loss", type=int, default=1)
    parser.add_argument("--e_loss", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.8)

    parser.add_argument("--shift", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--load_mode", type=int, default=1)

    parser.add_argument("--detach", type=int, default=1)
    parser.add_argument("--layer_depth", type=int, default=3)
    args = parser.parse_args()
    main()
