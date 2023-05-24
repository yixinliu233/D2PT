import arguments
import time

from utils import *
from models import DDPT
from early_stop import EarlyStopping, Stop_args
from load_data import load_data

args = arguments.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    data, meta = load_data(args.dataset, args.train_per_class, args.val_per_class,
                           args.missing_link, args.missing_feature,
                           normalize_features=args.normalize_features, ogb_train_ratio=args.ogb_train_ratio)

    x = data.x
    labels = data.y.to(device)
    idx_train = torch.where(data.train_mask==True)[0]
    idx_val = torch.where(data.val_mask == True)[0]
    idx_test = torch.where(data.test_mask == True)[0]
    num_nodes = data.num_nodes
    num_classes = meta['num_classes']

    # Compute x_prop
    adj = edge_index_to_sparse_mx(data.edge_index, num_nodes)
    adj = process_adj(adj)
    x_prop = feature_propagation(adj, x, args.T, args.alpha)

    # Compute x_prop_aug
    if args.lambda_ce_aug > 0 or args.lambda_pa > 0:
        if args.dataset == 'arxiv':
            adj_knn = get_knn_graph(x_prop, args.num_neighbor, knn_metric=args.knn_metric, batch_size=5000)
        else:
            adj_knn = get_knn_graph(x_prop, args.num_neighbor, knn_metric=args.knn_metric)
        adj_knn = process_adj(adj_knn)
        x_prop_aug = feature_propagation(adj_knn, x, args.T, args.alpha)

    model = DDPT(nfeat=x.shape[1],
                 nhid=args.hidden,
                 nclass=num_classes,
                 dropout=args.dropout,
                 use_bn = args.use_bn).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x_prop = x_prop.to(device)
    if args.lambda_ce_aug > 0 or args.lambda_pa > 0:
        x_prop_aug = x_prop_aug.to(device)
    else:
        x_prop_train = x_prop[idx_train]
        labels_train = labels[idx_train]

    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)

    def test(labels):
        model.eval()
        _, output = model(x_prop, True)

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        test_acc = accuracy(output[idx_test], labels[idx_test])

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(test_acc.item()),
              )
        return test_acc.item()

    def train_full_batch():
        feat, output = model(x_prop)
        ce_loss = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train = ce_loss

        feat_aug, output_aug = model(x_prop_aug)

        if args.lambda_ce_aug > 0:
            ce_loss_aug = F.nll_loss(output_aug[idx_train], labels[idx_train])
            loss_train += ce_loss_aug * args.lambda_ce_aug

        if args.lambda_pa > 0:
            output_exp = torch.exp(output)
            confidences = output_exp.max(1)[0]
            pseudo_labels = output_exp.max(1)[1].type_as(labels)
            pseudo_labels[idx_train] = labels[idx_train]
            confidences[idx_train] = 1.0

            proto_aug = get_proto_norm_weighted(num_classes, feat_aug, pseudo_labels, confidences)
            proto = get_proto_norm_weighted(num_classes, feat, pseudo_labels, confidences)

            loss_pa = proto_align_loss(proto_aug, proto)
            loss_train += loss_pa * args.lambda_pa

        model.zero_grad()
        loss_train.backward()
        optimizer.step()

    def train_mini_batch(batch_size):
        idx_not_train = torch.where(data.train_mask == False)[0]
        idx_unlabel = idx_not_train[torch.randperm(idx_not_train.size(0))][:batch_size]
        idx_batch = torch.cat((idx_train, idx_unlabel))
        num_train = idx_train.shape[0]

        feat, output = model(x_prop[idx_batch])
        ce_loss = F.nll_loss(output[:num_train], labels[idx_train])
        loss_train = ce_loss

        feat_aug, output_aug = model(x_prop_aug[idx_batch])

        if args.lambda_ce_aug > 0:
            ce_loss_aug = F.nll_loss(output_aug[:num_train], labels[idx_train])
            loss_train += ce_loss_aug * args.lambda_ce_aug

        if args.lambda_pa > 0:
            output_exp = torch.exp(output)
            confidences = output_exp.max(1)[0]
            pseudo_labels = output_exp.max(1)[1].type_as(labels)
            pseudo_labels[:num_train] = labels[idx_train]
            confidences[:num_train] = 1.0

            proto_aug = get_proto_norm_weighted(num_classes, feat_aug, pseudo_labels, confidences)
            proto = get_proto_norm_weighted(num_classes, feat, pseudo_labels, confidences)

            loss_pa = proto_align_loss(proto_aug, proto)
            loss_train += loss_pa * args.lambda_pa

        model.zero_grad()
        loss_train.backward()
        optimizer.step()

    def train_base():
        feat, output = model(x_prop_train)
        ce_loss = F.nll_loss(output, labels_train)
        loss_train = ce_loss

        model.zero_grad()
        loss_train.backward()
        optimizer.step()

    t = time.time()
    for epoch in range(args.epochs):
        if args.lambda_ce_aug > 0 or args.lambda_pa > 0:
            if args.batch_size == 0:
                train_full_batch()
            else:
                train_mini_batch(args.batch_size)
        else:
            train_base()

        model.eval()
        _, output = model(x_prop)

        loss_val = (F.nll_loss(output[idx_val], labels[idx_val])).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()
        if early_stopping.check([acc_val, loss_val], epoch):
            break
        acc_test = accuracy(output[idx_test], labels[idx_test]).item()

        if epoch % 50 == 0:
            current_time = time.time()
            tt = current_time - t
            t = current_time
            print('epoch:{} , acc_val:{:.4f} , acc_test:{:.4f}, time:{:.4f}s'.format(epoch, acc_val, acc_test, tt))

    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    model.load_state_dict(early_stopping.best_state)
    test_acc = test(labels)

    return test_acc

if __name__ == "__main__":

    accs = []
    for trial in range(1, args.num_trials + 1):
        setup_seed(trial)
        test_acc = main()
        print('Trial:{}, Test_acc:{:.4f}'.format(trial, test_acc))
        accs.append(test_acc)

    avg_acc = np.mean(accs) * 100
    std_acc = np.std(accs) * 100
    print('[FINAL RESULT] AVG_ACC:{:.2f}+-{:.2f}'.format(avg_acc, std_acc))