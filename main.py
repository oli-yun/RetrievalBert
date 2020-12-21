import torch
from tqdm import tqdm
import torch.nn as nn
import time
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from models import PreTrainModel, OnlyKNN, KNNBackoff, KNNStaticConcat, UpdateKNNAdaptiveConcat, \
    KNNBackoffTwoModels, KNNStaticConcatTwoModels
from knn import KNNDstore
from parameters import parse
from datasets import load_tokenized_dataset, BalancedBatchSampler, generate_dataloader, preprocess_data
from utils import EarlyStopping, SemihardNegativeTripletSelector, HardNegativePairSelector, set_seed, check_dir, \
    get_optimizer
from trainer import fit, test_epoch, no_args_train
from metrics import AccumulatedAccuracyMetric
from losses import OnlineTripletLoss, OnlineContrastiveLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    test_loss, metric = test_epoch(test_dataloader, model, criterion, device, metric)
    message = '**Test set: Average loss: {:.4f}\t{}: {}'.format(test_loss, metric.name(), metric.value())
    logger.success(message)

    logger.info('Generate datastore.')
    model.only_return_hidden_states = True
    model.only_return_logits = False
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


def fine_tune_with_knn(args, fixed_finetune=False, fixed_knn=False, only_knn=False):
    start_time = time.strftime("-%Y-%m-%d", time.localtime())
    logger.add(args.log_dir + 'update_dstore_adaptive_weight' + start_time + '.log')
    logger.info("Load data.")
    train_dataset = load_tokenized_dataset(args.tokenized_data_dir + 'train')
    _, train_dataloader, dev_dataloader, test_dataloader = generate_dataloader(args, with_idx=True)

    logger.info("Load model.")
    pretrain_model = PreTrainModel(args.pretrain_model_name, args.num_labels)
    pretrain_model.load_state_dict(torch.load(args.model_dir + f"{args.pretrain_model_name}.pt"))
    knn_store = KNNDstore(args)
    knn_store.read_dstore(args.dstore_dir + f'finetune/keys.npy', args.dstore_dir + f'finetune/vals.npy')
    knn_store.add_index()

    ordered_dataloader = None
    knn_embedding_model = None
    if fixed_knn:
        knn_embedding_model = PreTrainModel(args.pretrain_model_name, args.num_labels, only_return_hidden_states=True)
        knn_embedding_model.load_state_dict(torch.load(args.model_dir + f"{args.pretrain_model_name}.pt"))
        for param in knn_embedding_model.parameters():
            param.requires_grad = False
    elif not fixed_knn and not fixed_finetune:
        ordered_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    model = UpdateKNNAdaptiveConcat(pretrain_model, knn_store, args.k, args.temperature, train_dataset,
                                    knn_embedding_model, fixed_finetune, fixed_knn, only_knn)
    if fixed_finetune:
        model_path = args.model_dir + "fine_tune_with_knn(fixed_finetune).pt"
    elif fixed_knn:
        model_path = args.model_dir + "fine_tune_with_knn(fixed_knn).pt"
    elif only_knn:
        model_path = args.model_dir + "fine_tune_with_knn(only_knn).pt"
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

    model = model.to(device)
    if not fixed_knn:
        model.knn_store.load_best()

    model.start_test()
    test_loss, metric = test_epoch(test_dataloader, model, criterion, device, metric)
    message = '**Test set: Average loss: {:.4f}\t{}: {}'.format(test_loss, metric.name(), metric.value())
    logger.success(message)


def static_concat_models(args):
    logger.info('Load models')
    pretrain_model = PreTrainModel(args.pretrain_model_name, args.num_labels)
    pretrain_model.load_state_dict(torch.load(args.model_dir + f"{args.pretrain_model_name}.pt"))

    knn_store = KNNDstore(args)
    knn_store.load_best()

    knn_model = UpdateKNNAdaptiveConcat(PreTrainModel(args.pretrain_model_name, args.num_labels),
                                        knn_store, args.k, args.temperature, None,
                                        None, False, False, True)
    knn_model.load_state_dict(torch.load(args.model_dir + "fine_tune_with_knn(only_knn).pt"))

    logger.info('Load data')
    _, _, dev_dataloader, test_dataloader = generate_dataloader(args)

    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model = KNNStaticConcatTwoModels(pretrain_model, knn_model.pretrain_model, knn_store)
    model.to(device)

    dev_acc = [0 for _ in range(len(threshold))]
    cnt = 0
    for data, target in dev_dataloader:
        data = tuple(d.to(device) for d in data)
        cnt += target.size()[-1]
        for i, th in enumerate(threshold):
            input = data + (args.k, args.temperature, th)
            with torch.no_grad():
                prob = model(*input)
            dev_acc[i] += (torch.argmax(prob, dim=1) == target).sum().item()

    dev_acc = [acc / cnt for acc in dev_acc]
    print(dev_acc)


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
    set_seed(arg.seed)

    check_dir(arg)  # run at the very start
    # preprocess_data(arg)  # run at the very start
    # fine_tune_pretrain_model_generate_datastore(arg)
    # run_no_arg_knn(arg)
    # fine_tune_with_knn(arg)
    fine_tune_with_knn(arg, fixed_finetune=True)
    # fine_tune_with_knn(arg, fixed_knn=True)
    # fine_tune_with_knn(arg, only_knn=True)
    # metric_learning(arg, triplet=True)
    # metric_learning(arg, triplet=False)
    # static_concat_models(arg)
