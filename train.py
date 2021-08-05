import os
import time
import argparse
import configparser
import numpy as np
import torch
import torch.nn as nn
import tqdm

from engine import trainer
from utils import *
from model import STSGCN
import ast

DATASET = 'PEMSD4'  # PEMSD4 or PEMSD8

config_file = './{}.conf'.format(DATASET)
config = configparser.ConfigParser()
config.read(config_file)


parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--no_cuda', action="store_true", help="没有GPU")
parser.add_argument('--data', type=str, default=config['data']['data'], help='data path')
parser.add_argument('--sensors_distance', type=str, default=config['data']['sensors_distance'], help='节点距离文件')
parser.add_argument('--column_wise', type=eval, default=config['data']['column_wise'],
                    help='是指列元素的级别上进行归一，否则是全样本取值')
parser.add_argument('--normalizer', type=str, default=config['data']['normalizer'], help='归一化方式')
parser.add_argument('--batch_size', type=int, default=config['data']['batch_size'], help="batch大小")

parser.add_argument('--num_of_vertices', type=int, default=config['model']['num_of_vertices'], help='传感器数量')
parser.add_argument('--construct_type', type=str, default=config['model']['construct_type'],
                    help="构图方式  {connectivity, distance}")
parser.add_argument('--in_dim', type=int, default=config['model']['in_dim'], help='输入维度')
parser.add_argument('--hidden_dims', type=list, default=ast.literal_eval(config['model']['hidden_dims']),
                    help='中间各STSGCL层的卷积操作维度')
parser.add_argument('--first_layer_embedding_size', type=int, default=config['model']['first_layer_embedding_size'],
                    help='第一层输入层的维度')
parser.add_argument('--out_layer_dim', type=int, default=config['model']['out_layer_dim'], help='输出模块中间层维度')
parser.add_argument("--history", type=int, default=config['model']['history'], help="每个样本输入的离散时序")
parser.add_argument("--horizon", type=int, default=config['model']['horizon'], help="每个样本输出的离散时序")
parser.add_argument("--strides", type=int, default=config['model']['strides'], help="滑动窗口步长，local时空图使用几个时间步构建的，默认为3")
parser.add_argument("--temporal_emb", type=eval, default=config['model']['temporal_emb'], help="是否使用时间嵌入向量")
parser.add_argument("--spatial_emb", type=eval, default=config['model']['spatial_emb'], help="是否使用空间嵌入向量")
parser.add_argument("--use_mask", type=eval, default=config['model']['use_mask'], help="是否使用mask矩阵优化adj")
parser.add_argument("--activation", type=str, default=config['model']['activation'], help="激活函数 {relu, GlU}")

parser.add_argument('--seed', type=int, default=config['train']['seed'], help='种子设置')
parser.add_argument("--learning_rate", type=float, default=config['train']['learning_rate'], help="初始学习率")
parser.add_argument("--lr_decay", type=eval, default=config['train']['lr_decay'], help="是否开启初始学习率衰减策略")
parser.add_argument("--lr_decay_step", type=str, default=config['train']['lr_decay_step'], help="在几个epoch进行初始学习率衰减")
parser.add_argument("--lr_decay_rate", type=float, default=config['train']['lr_decay_rate'], help="学习率衰减率")
parser.add_argument('--epochs', type=int, default=config['train']['epochs'], help="训练代数")
parser.add_argument('--print_every', type=int, default=config['train']['print_every'], help='几个batch报训练损失')
parser.add_argument('--save', type=str, default=config['train']['save'], help='保存路径')
parser.add_argument('--expid', type=int, default=config['train']['expid'], help='实验 id')
parser.add_argument('--max_grad_norm', type=float, default=config['train']['max_grad_norm'], help="梯度阈值")

parser.add_argument('--patience', type=int, default=config['train']['patience'], help='等待代数')
parser.add_argument('--log_file', default=config['train']['log_file'], help='log file')

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


log = open(args.log_file, 'w')
log_string(log, str(args))


def main():
    # load data
    adj = get_adjacency_matrix(distance_df_filename=args.sensors_distance,
                               num_of_vertices=args.num_of_vertices,
                               type_=args.construct_type,
                               id_filename=None)
    local_adj = construct_adj(A=adj, steps=args.strides)
    local_adj = torch.FloatTensor(local_adj)

    dataloader = load_dataset(dataset_dir=args.data,
                              normalizer=args.normalizer,
                              batch_size=args.batch_size,
                              valid_batch_size=args.batch_size,
                              test_batch_size=args.batch_size,
                              column_wise=args.column_wise)

    scaler = dataloader['scaler']

    log_string(log, 'loading data...')

    log_string(log, "The shape of localized adjacency matrix: {}".format(local_adj.shape))

    log_string(log, f'trainX: {torch.tensor(dataloader["train_loader"].xs).shape}\t\t '
                    f'trainY: {torch.tensor(dataloader["train_loader"].ys).shape}')
    log_string(log, f'valX:   {torch.tensor(dataloader["val_loader"].xs).shape}\t\t'
                    f'valY:   {torch.tensor(dataloader["val_loader"].ys).shape}')
    log_string(log, f'testX:   {torch.tensor(dataloader["test_loader"].xs).shape}\t\t'
                    f'testY:   {torch.tensor(dataloader["test_loader"].ys).shape}')
    log_string(log, f'mean:   {scaler.mean:.4f}\t\tstd:   {scaler.std:.4f}')
    log_string(log, 'data loaded!')

    engine = trainer(args=args,
                     scaler=scaler,
                     adj=local_adj,
                     history=args.history,
                     num_of_vertices=args.num_of_vertices,
                     in_dim=args.in_dim,
                     hidden_dims=args.hidden_dims,
                     first_layer_embedding_size=args.first_layer_embedding_size,
                     out_layer_dim=args.out_layer_dim,
                     log=log,
                     lrate=args.learning_rate,
                     device=device,
                     activation=args.activation,
                     use_mask=args.use_mask,
                     max_grad_norm=args.max_grad_norm,
                     lr_decay=args.lr_decay,
                     temporal_emb=args.temporal_emb,
                     spatial_emb=args.spatial_emb,
                     horizon=args.horizon,
                     strides=args.strides)

    # 开始训练
    log_string(log, 'compiling model...')
    his_loss = []
    val_time = []
    train_time = []

    wait = 0
    val_loss_min = float('inf')
    best_model_wts = None

    for i in tqdm.tqdm(range(1, args.epochs + 1)):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {i:04d}')
            break

        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            # [B, T, N, C]

            trainy = torch.Tensor(y[:, :, :, 0]).to(device)
            # [B, T, N]

            loss, tmae, tmape, trmse = engine.train(trainx, trainy)
            train_loss.append(loss)
            train_mae.append(tmae)
            train_mape.append(tmape)
            train_rmse.append(trmse)

            if iter % args.print_every == 0:
                logs = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, lr: {}'
                print(logs.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1],
                                  engine.optimizer.param_groups[0]['lr']), flush=True)

        if args.lr_decay:
            engine.lr_scheduler.step()

        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            valx = torch.Tensor(x).to(device)
            # [B, T, N, C]

            valy = torch.Tensor(y[:, :, :, 0]).to(device)
            # [B, T, N]

            vmae, vmape, vrmse = engine.evel(valx, valy)
            valid_loss.append(vmae)
            valid_mape.append(vmape)
            valid_rmse.append(vrmse)

        s2 = time.time()
        logs = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        log_string(log, logs.format(i, (s2-s1)))

        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        logs = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        log_string(log, logs.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))
        # os.system('nvidia-smi')

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        if mvalid_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {mvalid_loss:.4f}, '
                f'save model to {args.save + "exp_" + str(args.expid) + "_" + str(round(mvalid_loss, 2)) + "_best_model.pth"}'
            )
            wait = 0
            val_loss_min = mvalid_loss
            best_model_wts = engine.model.state_dict()
            torch.save(best_model_wts,
                       args.save + "exp_" + str(args.expid) + "_" + str(round(val_loss_min, 2)) + "_best_model.pth")
        else:
            wait += 1

        np.save('./history_loss' + f'_{args.expid}', his_loss)

    log_string(log, "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    log_string(log, "Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # 测试
    engine.model.load_state_dict(
        torch.load(args.save + "exp_" + str(args.expid) + "_" + str(round(val_loss_min, 2)) + "_best_model.pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test'][:, :, :, 0]).to(device)
    # B, T, N

    for iter, (x, y) in tqdm.tqdm(enumerate(dataloader['test_loader'].get_iterator())):
        testx = torch.Tensor(x).to(device)
        with torch.no_grad():
            preds = engine.model(testx)
            # [B, T, N]
            outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]  # 这里是因为你在做batch的时候，可能会padding出新的sample以满足batch_size的要求

    log_string(log, "Training finished")
    log_string(log, "The valid loss on best model is " + str(round(val_loss_min, 4)))

    amae = []
    amape = []
    armse = []

    for t in range(args.horizon):
        pred = scaler.inverse_transform(yhat[:, t, :])
        real = realy[:, t, :]

        mae, mape, rmse = metric(pred, real)
        logs = '最好的验证模型在测试集上对 horizon: {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'

        log_string(log, logs.format(t+1, mae, mape, rmse))
        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)

    logs = '总平均测试结果, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    log_string(log, logs.format(np.mean(amae), np.mean(amape), np.mean(armse)))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()

    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()



