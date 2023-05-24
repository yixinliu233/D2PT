import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--missing_link', type=float, default=0.0)
    parser.add_argument('--missing_feature', type=float, default=0.0)
    parser.add_argument('--train_per_class', type=int, default=20)
    parser.add_argument('--val_per_class', type=int, default=30)
    parser.add_argument('--ogb_train_ratio', type=float, default=1.0)

    parser.add_argument('--num_trials', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--patience', type=int, default=2000)
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--normalize_features', type=bool, default=True)

    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=64)

    parser.add_argument('--lambda_pa', type=float, default=4)
    parser.add_argument('--lambda_ce_aug', type=float, default=0.2)
    parser.add_argument('--num_neighbor', type=int, default=5)
    parser.add_argument('--knn_metric', type=str, default='cosine', choices=['cosine','minkowski'])
    parser.add_argument('--batch_size', type=int, default=0)

    return parser.parse_args()