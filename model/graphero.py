from embedder import embedder
import torch.nn as nn
import layers
import torch.optim as optim
import utils
import torch.nn.functional as F
import torch
import numpy as np
from copy import deepcopy
import os


def kl_div(p, q):
    return (p * ((p + 1e-10) / (q + 1e-10)).log()).sum(dim=1)
def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

class graphero():
    def __init__(self, args):
        self.args = args

    def training(self):
        self.args.embedder = f'({self.args.layer.upper()})' + self.args.embedder + f'_noise_type_{self.args.noise_type}' + f'_noise_rate_{self.args.noise_rate}' + f'_imbalance_rate_{self.args.imbalance_ratio}'
        if self.args.im_ratio == 1: 
            os.makedirs(f'./results/baseline/natural/{self.args.dataset}', exist_ok=True)
            text = open(f'./results/baseline/natural/{self.args.dataset}/{self.args.embedder}.txt', 'w')
        else: 
            os.makedirs(f'./results/baseline/manual/{self.args.dataset}/{self.args.im_class_num}/{self.args.im_ratio}',
                        exist_ok=True)
            text = open(
                f'./results/baseline/manual/{self.args.dataset}/{self.args.im_class_num}/{self.args.im_ratio}/{self.args.embedder}.txt',
                'w')

        seed_result = {}
        seed_result['acc'] = []
        seed_result['macro_F'] = []
        seed_result['gmeans'] = []
        seed_result['bacc'] = []

        for seed in range(5, 5 + self.args.num_seed):
            print(f'============== seed:{seed} ==============')
            utils.seed_everything(seed)
            print('seed:', seed, file=text)
            self = embedder(self.args)

            model = modeler(self.args, self.adj).to(self.args.device)
            optimizer_fe = optim.Adam(model.encoder.parameters(), lr=self.args.lr,
                                      weight_decay=self.args.wd)  
            optimizer_cls = optim.Adam(model.classifier.parameters(), lr=self.args.lr,
                                       weight_decay=self.args.wd) 
            val_f = []
            test_results = []
            best_metric = 0
            for epoch in range(self.args.ep):
                model.train()
                optimizer_fe.zero_grad()
                optimizer_cls.zero_grad()
                loss = model(self.origin_features, self.features, self.labels, self.idx_train, epoch)
                loss.backward()
                optimizer_fe.step()
                optimizer_cls.step()
                model.eval()
                embed = model.encoder(self.origin_features, self.features)
                output = model.classifier(embed)
                acc_val, macro_F_val, gmeans_val, bacc_val = utils.performance_measure(output[self.idx_val],
                                                                                       self.labels[self.idx_val],
                                                                                       pre='valid')
                val_f.append(macro_F_val)
                max_idx = val_f.index(max(val_f))
                if best_metric <= macro_F_val:
                    best_metric = macro_F_val
                    best_model = deepcopy(model)
                acc_test, macro_F_test, gmeans_test, bacc_test = utils.performance_measure(output[self.idx_test],
                                                                                           self.labels[self.idx_test],
                                                                                           pre='test')
                test_results.append([acc_test, macro_F_test, gmeans_test, bacc_test])
                best_test_result = test_results[max_idx]
                st = "[seed {}][{}][Epoch {}]".format(seed, self.args.embedder, epoch)
                st += "[Val] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}|| ".format(acc_val,
                                                                                                     macro_F_val,
                                                                                                     gmeans_val,
                                                                                                     bacc_val)
                st += "[Test] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}\n".format(acc_test,
                                                                                                     macro_F_test,
                                                                                                     gmeans_test,
                                                                                                     bacc_test)
                st += "  [*Best Test Result*][Epoch {}] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}".format(
                    max_idx, best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3])
                if epoch % 100 == 0:
                    print(st)
                if (epoch - max_idx > self.args.ep_early) or (epoch + 1 == self.args.ep):
                    if epoch - max_idx > self.args.ep_early:
                        print("Early stop")
                    embed = best_model.encoder(self.origin_features, self.features)
                    output = best_model.classifier(embed)
                    best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[
                        3] = utils.performance_measure(output[self.idx_test], self.labels[self.idx_test], pre='test')
                    print("[Best Test Result] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}".format(
                        best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3]), file=text)
                    print(utils.classification(output[self.idx_test], self.labels[self.idx_test].detach().cpu()),
                          file=text)
                    print(utils.confusion(output[self.idx_test], self.labels[self.idx_test].detach().cpu()), file=text)
                    print(file=text)
                    break

            seed_result['acc'].append(float(best_test_result[0]))
            seed_result['macro_F'].append(float(best_test_result[1]))
            seed_result['gmeans'].append(float(best_test_result[2]))
            seed_result['bacc'].append(float(best_test_result[3]))

        acc = seed_result['acc']
        f1 = seed_result['macro_F']
        gm = seed_result['gmeans']
        bacc = seed_result['bacc']

        print(
            '[Averaged result] ACC: {:.1f}+{:.1f}, Macro-F: {:.1f}+{:.1f}, G-Means: {:.1f}+{:.1f}, bACC: {:.1f}+{:.1f}'.format(
                np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(gm), np.std(gm), np.mean(bacc),
                np.std(bacc)))
        print(file=text)
        print('ACC Macro-F G-Means bACC', file=text)
        print('{:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(f1),
                                                                               np.std(f1), np.mean(gm), np.std(gm),
                                                                               np.mean(bacc), np.std(bacc)), file=text)
        print(file=text)
        print(self.args, file=text)
        print(self.args)
        text.close()


class modeler(nn.Module):
    def __init__(self, args, adj):
        super(modeler, self).__init__()
        self.args = args
        self.encoder = layers.MLP_encoder(nfeat=args.nfeat, nhid=args.nhid, dropout=args.dropout)
        self.classifier = layers.MLP_classifier(nfeat=args.nhid, nclass=args.nclass, dropout=args.dropout)

    def forward(self, origin_features, features, labels, idx_train, epoch):
        embed = self.encoder(origin_features, features)
        output = self.classifier(embed)
        onehot_labels = F.one_hot(labels, num_classes=output.shape[1])
        pred_labels = F.softmax(output)
        jsd = js_div(pred_labels, onehot_labels)
        num_classes = output.shape[1]
        mean_jsd = torch.zeros(num_classes)
        std_jsd = torch.zeros(num_classes)
        mask = torch.zeros(len(jsd), dtype=torch.bool).to(labels.device)
        mask[idx_train] = True      
        for i in range(num_classes):
            class_mask = (labels == i) & mask
            class_jsd = jsd[class_mask]
            mean_jsd[i] = class_jsd.mean()
            std_jsd[i] = class_jsd.std()    
        k = 1
        thresholds = mean_jsd + k * std_jsd    
        f_train_idx = []
        for idx in idx_train:
            if jsd[idx] < thresholds[labels[idx]]:
                f_train_idx.append(idx)    
        f_train_idx = torch.tensor(f_train_idx).to(idx_train.device)
        weight = features.new((labels.max().item() + 1)).fill_(1)
        if self.args.im_ratio == 0.01 or self.args.im_ratio == 1:
            num_classes = len(set(labels.tolist()))
            cls_num_list = np.zeros((num_classes)).astype(int)
            for i in range(num_classes):
                c_idx = (labels[idx_train] == i).nonzero()[:, -1].tolist()
                import math
                cls_num_list[i] = len(c_idx)
                weight[i] = 1 / (len(c_idx) + 1)
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss_values = criterion(output[idx_train], labels[idx_train])
        weighted_loss = torch.gather(weight, 0, labels[idx_train])
        loss_values = loss_values * weighted_loss
        loss_nodeclassification = torch.sum(loss_values) / torch.sum(weighted_loss)
        if epoch > 15:
            loss_nodeclassification = torch.sum(loss_values) / torch.sum(weighted_loss)
        elif epoch < 2:
            cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
            iota_list = 0.75 * np.log(cls_probs)
            output = output + torch.tensor(iota_list, dtype=torch.float32).to(output.device)
            loss_nodeclassification = F.cross_entropy(output[f_train_idx], labels[f_train_idx])
        return loss_nodeclassification
