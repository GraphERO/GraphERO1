import scipy.sparse as sp
import numpy as np
import torch
import data_load
import utils
from torch_geometric.utils.loop import add_self_loops, remove_self_loops

class embedder:
    def __init__(self, args):
        if args.gpu == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        self._dataset = data_load.Dataset(root="data", dataset=args.dataset, is_normalize=args.is_normalize, add_self_loop=args.add_sl)
        self.edge_index = self._dataset.edge_index.to(args.device)
        adj = self._dataset.adj
        features = self._dataset.features
        self.origin_features = features.to(args.device)
        labels = self._dataset.labels
        if args.propagate == True:
            def feature_propagation(adj, features, k, alpha):
                features = features.to(args.device)
                adj = adj.to(args.device)
                features_prop = features.clone()
                for i in range(1, k + 1):
                    features_prop = torch.sparse.mm(adj, features_prop)
                    features_prop = (1 - alpha) * features_prop + alpha * features
                features_p = features_prop.cpu()
                del features_prop
                del adj
                del features
                torch.cuda.empty_cache()
                return features_p

            adj_p = utils.normalize_sym(adj) 
            features = feature_propagation(adj_p, features, 10, 0.1)
        class_sample_num = 20
        im_class_num = args.im_class_num

        if (args.im_ratio == 0.01 or args.im_ratio == 0.02) & (args.dataset in ['Cora', 'CiteSeer']):
                args.criterion = 'mean'
                data = self._dataset
                n_cls = labels.max().item() + 1
                data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
                stats = labels[data_train_mask]
                n_data = []
                for i in range(n_cls):
                    data_num = (stats == i).sum()
                    n_data.append(int(data_num.item()))
                class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = utils.make_longtailed_data_remove(
                    data.edge_index, labels, n_data, n_cls, args.imbalance_ratio, data_train_mask.clone())
                edge = data.edge_index[:,train_edge_mask]              
                if args.add_sl:
                    edge = remove_self_loops(edge)[0]
                    edge = add_self_loops(edge)[0]
                adj = sp.coo_matrix((np.ones(edge.shape[1]), (edge[0,:], edge[1,:])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
                adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
                labels, og_to_new = utils.refine_label_order(labels)
                if args.dataset=='Cora': 
                    labels[labels==1] = 100
                    labels[labels==2] = 1
                    labels[labels==100] = 2
                elif args.dataset=='CiteSeer':  
                    labels[labels==0] = 100
                    labels[labels==1] = 0
                    labels[labels==100] = 1
                total_nodes = len(labels)                
                idx_train = torch.tensor(range(total_nodes))[data_train_mask]
                idx_val = torch.tensor(range(total_nodes))[data_val_mask]
                idx_test = torch.tensor(range(total_nodes))[data_test_mask]
                idx_train, idx_val, idx_test, class_num_mat = utils.split_manual_lt(labels, idx_train, idx_val, idx_test)
                samples_per_label = torch.tensor(class_num_mat[:,0])
                if args.noise == True:
                    from utils import noisify_with_P
                    noise_rate = args.noise_rate
                    nclass = labels.max() + 1
                    train_labels = labels.numpy()[idx_train]
                    noise_y, P = noisify_with_P(train_labels, nclass, noise_rate, 10, args.noise_type) 
                    noise_labels = labels.numpy().copy()
                    noise_labels[idx_train] = noise_y
                    labels = torch.LongTensor(noise_labels)
        adj = utils.normalize_adj(adj) if args.adj_norm_1 else utils.normalize_sym(adj)
        self.adj = adj.to(args.device)
        self.features = features.to(args.device)
        self.labels = labels.to(args.device)
        self.class_sample_num = class_sample_num
        self.im_class_num = im_class_num
        self.idx_train = idx_train.to(args.device)
        self.idx_val = idx_val.to(args.device)
        self.idx_test = idx_test.to(args.device)
        self.samples_per_label = samples_per_label
        self.class_num_mat = class_num_mat
        args.nfeat = features.shape[1]
        args.nclass = labels.max().item() + 1
        args.im_class_num = im_class_num
        self.args = args
