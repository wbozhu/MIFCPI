# -*- coding: utf-8 -*-

import random
import os
from model import CPIMIF
from dataset import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter
from pytorchtools import EarlyStopping  
import timeit
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from processing import process
from utils import *
import math
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
import copy
from sklearn.model_selection import StratifiedKFold
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def show_result(DATASET,lable,Accuracy_List,Precision_List,Recall_List,AUC_List,AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    print("The {} model's results:".format(lable))
    with open("./{}/results.txt".format(DATASET), 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))

def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]

def t1_precess(model,data_test,LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        '''data preparation '''
        for i in tqdm(range(math.ceil(len(data_test[0]) / batch_size))):
            batch_data = [data_test[di][i * batch_size: (i + 1) * batch_size] for di in range(len(data_test))]
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad, amino_mask, label = batch2tensor(batch_data, 'cuda:0')
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps)
            loss = Loss(pred.float(), label.view(label.shape[0]).long())

            predicted_scores = F.softmax(pred, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            correct_labels = label.cpu().numpy().reshape(-1).tolist()

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Reacll = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)  
    return Y, P, test_loss, Accuracy, Precision, Reacll, AUC, PRC

def t1_model(model_max,dataset_load,save_path,DATASET, LOSS, dataset = "Train",lable = "best",save = True):

    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        t1_precess(model_max,dataset_load, LOSS)
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET,dataset,lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(lable, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print('best result:', results)
    return results,Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test


def get_kfold_data(i, datasets, k=5):
    
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset

def get_KFold_index(datasets,k):
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    compounds = datasets[0]
    interactions = datasets[4]
    train_data_5index, dev_data_5index = [], []
    for tra_idx, tes_idx in skf.split(compounds, interactions):
        train_data_5index.append(tra_idx)
        dev_data_5index.append(tes_idx)
    return train_data_5index, dev_data_5index
def get_iFold_data(datasets,tra_idx,tes_idx):
    compounds = datasets[0]
    adjacencies = datasets[1]
    fps = datasets[2]
    proteins = datasets[3]
    interactions = datasets[4]
    train_data, dev_data = [], []
    train_data.append([compounds[tra_idx], adjacencies[tra_idx], fps[tra_idx], proteins[tra_idx], interactions[tra_idx]])
    dev_data.append([compounds[tes_idx], adjacencies[tes_idx], fps[tes_idx], proteins[tes_idx], interactions[tes_idx]])
    return train_data[0], dev_data[0]

def split_total_data(datasets, ratio=0.8):
    n_pos, n_neg, train_pos, train_neg, test_pos,test_neg = 0, 0, 0, 0, 0, 0
    tra_idx, tes_idx = [], []
    train_data, test_data = [], []

    compounds,adjacencies,fps,proteins,interactions= datasets[0],datasets[1],datasets[2],datasets[3],datasets[4]
    for x in interactions:
        if x == 1:
            n_pos += 1
        else:
            n_neg += 1
    train_pos_n = int(n_pos * ratio)
    train_neg_n = int(n_neg * ratio)
    # train_total = train_pos_n + train_neg_n

    for i in range(len(interactions)):
        if interactions[i] == 1:  # pos
            if train_pos < train_pos_n:
                tra_idx.append(i)
                train_pos+=1
            else:
                tes_idx.append(i)
                test_pos+=1
        else:                     # neg
            if train_neg < train_neg_n:
                tra_idx.append(i)
                train_neg+=1
            else:
                tes_idx.append(i)
                test_neg+=1
    print('train_pos={},train_neg={},train_total={}'.format(train_pos, train_neg,len(tra_idx)))
    print('test_pos={},test_neg={},test_total={}'.format(test_pos, test_neg, len(tes_idx)))
    train_data.append(compounds[tra_idx])
    train_data.append(adjacencies[tra_idx])
    train_data.append(fps[tra_idx])
    train_data.append(proteins[tra_idx])
    train_data.append(interactions[tra_idx])

    test_data.append(compounds[tes_idx])
    test_data.append(adjacencies[tes_idx])
    test_data.append(fps[tes_idx])
    test_data.append(proteins[tes_idx])
    test_data.append(interactions[tes_idx])
    return train_data, test_data

def get_kfold_data1(i, datasets, k=5):
    fold_size = len(datasets[0]) // k
    validset=[]
    trainset=[]
    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        for x in range(len(datasets)):
            validset.append(datasets[x][val_start:val_end])
            c = np.concatenate((datasets[x][0:val_start], datasets[x][val_end:]), axis=0)
            trainset.append(c)
    elif i == 0:
        val_end = fold_size
        for x in range(len(datasets)):
            validset.append(datasets[x][val_start:val_end])
            trainset.append(datasets[x][val_end:])
    else:
        for x in range(len(datasets)):
            validset.append(datasets[x][val_start:])
            trainset.append(datasets[x][0:val_start])
    return trainset, validset
def get_TV_data(datasets,ratio=5):
    fold_size = len(datasets[0])//5
    validset = []
    trainset = []
    val_end = fold_size
    for x in range(len(datasets)):
        validset.append(datasets[x][0:val_end])
        trainset.append(datasets[x][val_end:])
    return trainset, validset

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    """select seed"""
    SEED = 2023
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    """init hyperparameters"""
    hp = hyperparameter()

    """Load preprocessed data."""
    DATASET = "human"   # "celegans"
    print("Train in " + DATASET)
    weight_CE = torch.FloatTensor([0.16,0.84]).cuda()
    data_dir = './datasets/' + DATASET
    if not os.path.isdir(data_dir):
        process(DATASET)
    dataset1 = load_data(data_dir,target_type=0)
    K_Fold = 5
    device = torch.device("cuda:0")
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []
    total_train_dataset, total_test_dataset = split_total_data(dataset1, 0.8)
    train_5index, dev_5index = get_KFold_index(total_train_dataset, K_Fold)
    for i_fold in range(K_Fold):
        print('*' * 30, 'No.', i_fold + 1, '-fold', '*' * 30)
        model_dir = './result_model/' + DATASET + '/'
        os.makedirs(model_dir, exist_ok=True)
        model_dir = model_dir + str(i_fold)+'_fold_best_model.pth'
        train_dataset, dev_dataset = get_iFold_data(total_train_dataset,train_5index[i_fold], dev_5index[i_fold])
        test_dataset = total_test_dataset
        valid_size = len(dev_dataset[0])
        train_size = len(train_dataset[0])

        """ create model"""
        atom_dict = pickle.load(open(data_dir + '/atom_dict', 'rb'))
        amino_dict = pickle.load(open(data_dir + '/amino_dict', 'rb'))

        model = CPIMIF(hp, len(atom_dict), len(amino_dict))
        if torch.cuda.is_available():
            model.cuda()
        model_max = copy.deepcopy(model)
        """weight initialize"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)
        
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)
        Loss = nn.CrossEntropyLoss(weight=weight_CE)
        # print(model)
        
        save_path = "./" + DATASET + "/{}".format(i_fold)
        note = ''
        writer = SummaryWriter(log_dir=save_path, comment=note)

        """Output files."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path+'The_results_of_whole_dataset.txt'

        with open(file_results, 'w') as f:
            hp_attr = '\n'.join(['%s:%s' % item for item in hp.__dict__.items()])
            f.write(hp_attr + '\n')

        early_stopping = EarlyStopping(savepath = save_path,patience=hp.Patience, verbose=True, delta=0)
        batch_size = hp.Batch_size

        """Start training."""
        print('Training...')
        start = timeit.default_timer()
        idx = np.arange(len(train_dataset[0]))
        max_PRC_dev = 0
        epoch_label = 0

        for epoch in range(1, hp.Epoch + 1):
            """train"""
            train_losses_in_epoch = []
            np.random.shuffle(idx)
            model.train()

            '''data preparation '''
            for i in tqdm(range(math.ceil(len(train_dataset[0]) / batch_size))):
                batch_data = [train_dataset[di][idx[i * batch_size: (i + 1) * batch_size]] for di in range(len(train_dataset))]
                atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad, amino_mask, label = batch2tensor(
                    batch_data, 'cuda:0')

                optimizer.zero_grad()
                pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps)
                train_loss = Loss(pred.float(), label.view(label.shape[0]).long())
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()

                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)
            writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)

            """valid"""
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for i in tqdm(range(math.ceil(len(dev_dataset[0]) / batch_size))):
                    batch_data = [dev_dataset[di][i * batch_size: (i + 1) * batch_size] for di in range(len(dev_dataset))]
                    atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad, amino_mask, label = batch2tensor(
                        batch_data, 'cuda:0')
                    valid_scores = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps)
                    valid_loss = Loss(valid_scores.float(), label.view(label.shape[0]).long())

                    valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                    valid_labels = label.cpu().numpy().reshape(-1).tolist()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    valid_losses_in_epoch.append(valid_loss.item())
                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)
            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            if PRC_dev > max_PRC_dev:
                print('Update the best AUPR_dev:', PRC_dev)
                print('Update the best model...')
                torch.save(model.state_dict(), model_dir)
                model_max = copy.deepcopy(model)
                max_PRC_dev = PRC_dev
                epoch_label = epoch
                print('Update the best epoch:', epoch_label)

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} '
                         )

            writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
            writer.add_scalar('Valid AUC', AUC_dev, epoch)
            writer.add_scalar('Valid AUPR', PRC_dev, epoch)
            writer.add_scalar('Valid Accuracy', Accuracy_dev, epoch)
            writer.add_scalar('Valid Precision', Precision_dev, epoch)
            writer.add_scalar('Valid Reacll', Reacll_dev, epoch)
            writer.add_scalar('Learn Rate', optimizer.param_groups[0]['lr'], epoch)

            print(print_msg)
            # early_stopping
            early_stopping(valid_loss_a_epoch, model, epoch)
            if early_stopping.early_stop:
                print("Early stopping.")
                break
        print("The best model is epoch", epoch_label)
        """test"""
        testset_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
            t1_model(model_max, test_dataset, save_path, DATASET, Loss, dataset="Test", lable="stable")
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        with open(save_path + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(testset_stable_results + '\n')

    show_result(DATASET, "stable",
                Accuracy_List_stable, Precision_List_stable, Recall_List_stable,
                AUC_List_stable, AUPR_List_stable)



