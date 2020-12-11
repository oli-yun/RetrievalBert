import os
import torch
from tqdm import tqdm
import torch.nn as nn
import time
import numpy as np
from loguru import logger
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from models import PreTrainModel, OnlyKNN, KNNBackoff, KNNStaticConcat, UpdateKNNAdaptiveConcat
from knn import KNNDstore
from parameters import parse
from datasets import load_sst2, SST2Dataset, save_tokenized_dataset, load_tokenized_dataset, BalancedBatchSampler
from utils import EarlyStopping, SemihardNegativeTripletSelector, HardNegativePairSelector
from trainer import fit, test_epoch, no_args_train
from metrics import AccumulatedAccuracyMetric
from losses import OnlineTripletLoss, OnlineContrastiveLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def check_dir(args):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.dstore_dir):
        os.mkdir(args.dstore_dir)
    if not os.path.exists(args.dstore_dir + 'finetune/'):
        os.mkdir(args.dstore_dir + 'finetune')
    if not os.path.exists(args.faiss_dir):
        os.mkdir(args.faiss_dir)
    if not os.path.exists(args.tokenized_data_dir):
        os.mkdir(args.tokenized_data_dir)


def preprocess_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_name)

    train_data, train_labels = load_sst2(args.data_dir + args.train_path)
    dev_data, dev_labels = load_sst2(args.data_dir + args.dev_path)
    test_data, test_labels = load_sst2(args.data_dir + args.test_path)

    train_dataset = SST2Dataset(train_data, train_labels, tokenizer)
    save_tokenized_dataset(args.tokenized_data_dir + 'train', train_dataset)
    dev_dataset = SST2Dataset(dev_data, dev_labels, tokenizer)
    save_tokenized_dataset(args.tokenized_data_dir + 'dev', dev_dataset)
    test_dataset = SST2Dataset(test_data, test_labels, tokenizer)
    save_tokenized_dataset(args.tokenized_data_dir + 'test', test_dataset)

    train_dataset_with_idx = SST2Dataset(train_data, train_labels, tokenizer, return_idx=True)
    save_tokenized_dataset(args.tokenized_data_dir + 'train_with_idx', train_dataset_with_idx)
    dev_dataset_with_idx = SST2Dataset(dev_data, dev_labels, tokenizer, return_idx=True)
    save_tokenized_dataset(args.tokenized_data_dir + 'dev_with_idx', dev_dataset_with_idx)
    test_dataset_with_idx = SST2Dataset(test_data, test_labels, tokenizer, return_idx=True)
    save_tokenized_dataset(args.tokenized_data_dir + 'test_with_idx', test_dataset_with_idx)


def generate_dataloader(args, with_idx=False):
    train_path = args.tokenized_data_dir + 'train_with_idx' if with_idx else args.tokenized_data_dir + 'train'
    dev_path = args.tokenized_data_dir + 'dev_with_idx' if with_idx else args.tokenized_data_dir + 'dev'
    test_path = args.tokenized_data_dir + 'test_with_idx' if with_idx else args.tokenized_data_dir + 'test'

    train_dataset, dev_dataset, test_dataset = \
        load_tokenized_dataset(train_path), load_tokenized_dataset(dev_path), load_tokenized_dataset(test_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataset, train_dataloader, dev_dataloader, test_dataloader


def get_optimizer(args, model, train_dataloader):
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.1)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.06 * total_steps, num_training_steps=total_steps)

    return optimizer, scheduler


def fine_tune_pretrain_model_generate_datastore(args):
    start_time = time.strftime("-%Y-%m-%d", time.localtime())
    logger.add(args.log_dir + 'finetune' + start_time + '.log')
    logger.info("Fine-tune " + args.pretrain_model_name)
    logger.info("Load model.")
    model = PreTrainModel(args.pretrain_model_name, num_labels=args.num_labels, only_return_logits=True)
    model_path = args.model_dir + f"{args.pretrain_model_name}.pt"

    logger.info("Load data.")
    train_dataset, train_dataloader, dev_dataloader, test_dataloader = generate_dataloader(args)

    metric = AccumulatedAccuracyMetric()
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer(args, model, train_dataloader)
    early_stopping = EarlyStopping(output_path=model_path, patience=3, compare_loss=False)

    logger.info('Start training.')
    fit(train_dataloader, dev_dataloader, model, criterion, optimizer, scheduler, early_stopping,
        args.epochs, device, metric)

    logger.info('Test model with best performance')
    model.load_state_dict(torch.load(model_path))
    test_loss, metric = test_epoch(test_dataloader, model, criterion, device, metric)
    message = '**Test set: Average loss: {:.4f}\t{}: {}'.format(test_loss, metric.name(), metric.value())
    logger.success(message)

    logger.info('Generate datastore.')
    model = PreTrainModel(args.pretrain_model_name, num_labels=args.num_labels, only_return_hidden_states=True) \
        .to(device)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    keys = np.memmap(args.dstore_dir + f'finetune/keys.npy', dtype=np.float32, mode='w+',
                     shape=(args.dstore_size, 768))
    vals = np.memmap(args.dstore_dir + f'finetune/vals.npy', dtype=np.int, mode='w+',
                     shape=(args.dstore_size, 1))
    for i, (data, target) in enumerate(tqdm(train_dataloader)):
        b_data = tuple(d.to(device) for d in data)
        b_labels = target.unsqueeze(dim=1)

        hidden_states = model(*b_data)
        keys[i * args.batch_size:(i + 1) * args.batch_size] = hidden_states.cpu().detach().numpy()
        vals[i * args.batch_size:(i + 1) * args.batch_size] = b_labels.numpy()
    logger.success("Done!")


def run_no_arg_knn(args):
    start_time = time.strftime("-%Y-%m-%d", time.localtime())
    logger.add(args.log_dir + 'knn_baselines' + start_time + '.log')

    pretrain_model = PreTrainModel(args.pretrain_model_name, num_labels=args.num_labels)
    pretrain_model.load_state_dict(torch.load(args.model_dir + f"{args.pretrain_model_name}.pt"))
    knn_store = KNNDstore(args)
    knn_store.read_dstore(args.dstore_dir + f'finetune/keys.npy', args.dstore_dir + f'finetune/vals.npy')
    knn_store.add_index()

    logger.info('Load data.')
    _, _, dev_dataloader, test_dataloader = generate_dataloader(args)

    logger.info('Find best hyperparameters.')
    candidate_k = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    candidate_tmp = [1, 10, 100]

    if args.no_arg_way == 'knn_only':
        logger.info("Evaluate knn_only")
        model = OnlyKNN(pretrain_model, knn_store)
        no_args_train(dev_dataloader, test_dataloader, model, device, candidate_k, candidate_tmp)
    elif args.no_arg_way == 'knn_backoff':
        candidate_threshold = [0.5, 0.6, 0.7, 0.8, 0.9]
        logger.info("Evaluate knn_backoff")
        model = KNNBackoff(pretrain_model, knn_store)
        no_args_train(dev_dataloader, test_dataloader, model, device, candidate_k, candidate_tmp, candidate_threshold)
    else:
        candidate_weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        logger.info("Evaluate static_weighted_concat")
        model = KNNStaticConcat(pretrain_model, knn_store)
        no_args_train(dev_dataloader, test_dataloader, model, device, candidate_k, candidate_tmp, candidate_weight)
    logger.success("Done!")


def fine_tune_with_knn(args, fixed_finetune=False, knn_store_file=None):
    start_time = time.strftime("-%Y-%m-%d", time.localtime())
    logger.add(args.log_dir + 'update_dstore_adaptive_weight' + start_time + '.log')
    logger.info("Load data.")
    train_dataset = load_tokenized_dataset(args.tokenized_data_dir + 'train')
    _, train_dataloader, dev_dataloader, test_dataloader = generate_dataloader(args, with_idx=True)

    logger.info("Load model.")
    pretrain_model = PreTrainModel(args.pretrain_model_name, args.num_labels, only_return_hidden_states=True)
    knn_store = KNNDstore(args)
    fixed_knn = False if knn_store_file is None else True
    ordered_dataloader = None
    if fixed_finetune:
        pretrain_model.load_state_dict(torch.load(args.model_dir + f"{args.pretrain_model_name}.pt"))
        knn_store.read_dstore(args.dstore_dir + f'finetune/keys.npy', args.dstore_dir + f'finetune/vals.npy')
        knn_store.add_index()
    elif fixed_knn:
        knn_store.read_dstore(knn_store_file[0], knn_store_file[1])
        knn_store.read_index(knn_store_file[2])
    else:
        pretrain_model.to(device)
        ordered_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        logger.info('Generate Datastore.')
        for i, (data, target) in enumerate(tqdm(ordered_dataloader)):
            with torch.no_grad():
                data = tuple(d.to(device) for d in data)
                target = target.unsqueeze(-1)
                output = pretrain_model(*data)
                knn_store.write_dstore(output, target, i * args.batch_size, args.batch_size)
        knn_store.add_index()

    model = UpdateKNNAdaptiveConcat(pretrain_model, knn_store, args.num_labels, args.k, args.temperature,
                                    train_dataset, fixed_finetune, fixed_knn)
    if fixed_finetune:
        model_path = args.model_dir + "fine_tune_with_knn(fixed_finetune).pt"
    elif fixed_knn:
        model_path = args.model_dir + "fine_tune_with_knn(fixed_knn).pt"
    else:
        model_path = args.model_dir + "fine_tune_with_knn.pt"

    metric = AccumulatedAccuracyMetric()
    criterion = nn.NLLLoss()
    optimizer, scheduler = get_optimizer(args, model, train_dataloader)
    early_stopping = EarlyStopping(output_path=model_path, patience=3, compare_loss=False, save_knn=True)

    logger.info('Start training.')
    fit(train_dataloader, dev_dataloader, model, criterion, optimizer, scheduler, early_stopping,
        args.epochs, device, metric, ordered_dataloader=ordered_dataloader)

    logger.info('Test model with best performance')
    model.load_state_dict(torch.load(model_path))
    # model.load_state_dict(torch.load(args.model_dir + "fine_tune_with_knn(fixed_finetune)-k4.pt"))

    model = model.to(device)
    model.knn_store.load_best()
    # knn_store_file = [arg.dstore_dir + 'finetune_with_knn(fixed_finetune)-k4/keys.npy.best',
    #                   arg.dstore_dir + 'finetune_with_knn(fixed_finetune)-k4/vals.npy.best',
    #                   arg.faiss_dir + 'finetune_with_knn(fixed_finetune)-k4/index.best']
    # model.knn_store.read_dstore(knn_store_file[0], knn_store_file[1])
    # model.knn_store.read_index(knn_store_file[2])

    model.start_test()
    test_loss, metric = test_epoch(test_dataloader, model, criterion, device, metric)
    message = '**Test set: Average loss: {:.4f}\t{}: {}'.format(test_loss, metric.name(), metric.value())
    logger.success(message)


def metric_learning(args, triplet=False):
    start_time = time.strftime("-%Y-%m-%d", time.localtime())
    logger.add(args.log_dir + 'metric_learning' + start_time + '.log')
    if triplet:
        logger.info('Triplet Net')
    else:
        logger.info('Siamese Net')

    logger.info('Load and tokenized data.')
    train_dataset, dev_dataset, test_dataset = load_tokenized_dataset(args.tokenized_data_dir + 'train'), \
                                               load_tokenized_dataset(args.tokenized_data_dir + 'dev'), \
                                               load_tokenized_dataset(args.tokenized_data_dir + 'test')
    train_batch_sampler = BalancedBatchSampler(train_dataset.labels, n_classes=args.num_labels,
                                               n_samples=args.batch_size // args.num_labels)
    dev_batch_sampler = BalancedBatchSampler(dev_dataset.labels, n_classes=args.num_labels,
                                             n_samples=args.batch_size // args.num_labels)

    logger.info('Build dataloader.')
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    online_train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    online_dev_loader = DataLoader(dev_dataset, batch_sampler=dev_batch_sampler, **kwargs)

    logger.info('Build model.')
    margin = 1
    model = PreTrainModel(args.pretrain_model_name, args.num_labels, only_return_hidden_states=True)
    model_path = args.model_dir + "triplet_model.pt" if triplet else args.model_dir + "siamese_model.pt"

    if triplet:
        criterion = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
    else:
        criterion = OnlineContrastiveLoss(margin, HardNegativePairSelector())
    optimizer, scheduler = get_optimizer(args, model, online_train_loader)
    early_stopping = EarlyStopping(output_path=model_path, patience=3)

    logger.info('Start training.')
    fit(online_train_loader, online_dev_loader, model, criterion, optimizer, scheduler, early_stopping,
        args.epochs, device, log_interval=100)

    logger.info('Start testing.')
    model.load_state_dict(torch.load(model_path))
    knn_store = KNNDstore(args)
    model.to(device)
    ordered_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info('Generate Datastore.')
    for i, (data, target) in enumerate(tqdm(ordered_dataloader)):
        with torch.no_grad():
            data = tuple(d.to(device) for d in data)
            target = target.unsqueeze(-1)
            output = model(*data)
            knn_store.write_dstore(output, target, i * args.batch_size, args.batch_size)
    knn_store.add_index()

    _, _, dev_dataloader, test_dataloader = generate_dataloader(args)

    logger.info('Find best hyperparameters.')
    candidate_k = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    candidate_tmp = [1, 10, 100]

    logger.info("Evaluate knn_only")
    model.only_return_hidden_states = False
    knn_model = OnlyKNN(model, knn_store)
    no_args_train(dev_dataloader, test_dataloader, knn_model, device, candidate_k, candidate_tmp)
    logger.success("Done!")


if __name__ == '__main__':
    arg = parse()
    # check_dir(arg)  # run at the very start
    # preprocess_data(arg)  # run at the very start
    # fine_tune_pretrain_model_generate_datastore(arg)
    # run_no_arg_knn(arg)
    # fine_tune_with_knn(arg)
    fine_tune_with_knn(arg, fixed_finetune=True)
    # fine_tune_with_knn(arg, knn_store_file=[arg.dstore_dir + 'finetune/keys.npy',
    #                                         arg.dstore_dir + 'finetune/vals.npy', arg.faiss_dir + 'finetune/index'])
    # metric_learning(arg, triplet=True)
