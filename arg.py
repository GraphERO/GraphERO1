import argparse

def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphero', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0, help="Choose GPU number")
    parser.add_argument('--dataset', type=str, default='cora_full', choices=['Cora', 'CiteSeer'])
    parser.add_argument('--im_class_num', type=int, default=3, help="Number of tail classes")
    parser.add_argument('--im_ratio', type=float, default=1, help="1 for natural, [0.2, 0.1, 0.05] for manual, 0.01 for LT setting")
    parser.add_argument('--layer', type=str, default='gcn', choices=['gcn', 'gat'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rw', type=float, default=0.000001, help="Balances edge loss within node classification loss")
    parser.add_argument('--ep', type=int, default=10000, help="Number of epochs to train.")
    parser.add_argument('--ep_early', type=int, default=1000, help="Early stop criterion.")
    parser.add_argument('--add_sl', type=str2bool, default=True, help="Whether to include self-loop")
    parser.add_argument('--adj_norm_1', action='store_false', default=True, help="D^(-1)A")
    parser.add_argument('--adj_norm_2', action='store_true', default=False, help="D^(-1/2)AD^(-1/2)")
    parser.add_argument('--nhid', type=int, default=64, help="Number of hidden dimensions")
    parser.add_argument('--nhead', type=int, default=1, help="Number of multi-heads")
    parser.add_argument('--wd', type=float, default=5e-4, help="Controls weight decay")
    parser.add_argument('--num_seed', type=int, default=3, help="Number of total seeds") 
    parser.add_argument('--is_normalize', action='store_true', default=False, help="Normalize features")
    parser.add_argument('--cls_og', type=str, default='GNN', choices=['GNN', 'MLP'], help="Wheter to user (GNN+MLP) or (MLP) as a classifier")
    parser.add_argument('--noise', type=str2bool, default=True, help="Wheter to apply Label Noise.")
    parser.add_argument('--noise_rate', type=float, default=0.4, help="Label Noise Rate.")
    parser.add_argument('--noise_type', type=str, default='pair', choices=['uniform', 'pair'], help="Label Noise Type")
    parser.add_argument('--propagate', type=str2bool, default=False, help="Wheter to use raw feature or propagated feature.")
    parser.add_argument('--imbalance_ratio', type=float, default=50, help="Control the LT setting/ 100=0.01  50=0.02  20=0.05")
    if parser.parse_known_args()[0].graphero:
        parser.add_argument('--embedder', nargs='?', default='graphero')
    return parser