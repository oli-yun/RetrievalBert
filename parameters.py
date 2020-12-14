import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model_name', type=str, default='roberta-base',
                        choices=['bert-base-uncased', 'roberta-base'])
    parser.add_argument('--no_arg_way', type=str, default='knn_only',
                        choices=['knn_only', 'knn_backoff', 'static_weighted_concat'])
    parser.add_argument('--test_no_arg', action='store_true')

    parser.add_argument('--dataset', type=str, default='SST2', choices=['SST2, ag_news'])
    parser.add_argument('--model_dir', type=str, default='./models/')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--dstore_dir', type=str, default='./dstore/')
    parser.add_argument('--faiss_dir', type=str, default='./faiss/')
    parser.add_argument('--data_dir', type=str, default='./sst2/')
    parser.add_argument('--tokenized_data_dir', type=str, default='./sst2/tokenized/')
    parser.add_argument('--train_path', type=str, default='train.tsv')
    parser.add_argument('--test_path', type=str, default='test.tsv')
    parser.add_argument('--dev_path', type=str, default='dev.tsv')

    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=20)  # 10
    parser.add_argument('--batch_size', type=int, default=16)  # 16/128
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--clusters_num', type=int, default=4)
    parser.add_argument('--nprobe', type=int, default=1)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--temperature', type=int, default=1)
    parser.add_argument('--knn_weight', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--dstore_size', type=int, default=67349)

    args = parser.parse_args()
    return args