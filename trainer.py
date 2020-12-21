import torch
import numpy as np
from loguru import logger
from tqdm import tqdm


def train_epoch(train_dataloader, model, loss_fn, optimizer, scheduler, device, metric=None, log_interval=1000):
    if metric is not None:
        metric.reset()
    model.train()
    losses = []
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_dataloader)):
        if type(data) not in (tuple, list):
            data = (data,)
        data = tuple(d.to(device) for d in data)
        target = target.to(device)

        model.zero_grad()
        outputs = model(*data)
        loss_inputs = (outputs,) if type(outputs) not in (tuple, list) else outputs
        loss_inputs += (target,)
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if metric is not None:
            metric(outputs, target)

        if (batch_idx + 1) % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_dataloader.dataset),
                100. * batch_idx / len(train_dataloader), np.mean(losses))
            if metric is not None:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            logger.info(message)
            losses = []

    total_loss /= len(train_dataloader)
    return total_loss, metric


def test_epoch(dataloader, model, loss_fn, device, metric=None):
    with torch.no_grad():
        if metric is not None:
            metric.reset()
        model.eval()
        val_loss = 0
        for data, target in dataloader:
            if not type(data) in (tuple, list):
                data = (data,)
            data = tuple(d.to(device) for d in data)
            target = target.to(device)
            outputs = model(*data)

            loss_inputs = (outputs,) if type(outputs) not in (tuple, list) else outputs
            loss_inputs += (target,)

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            if metric is not None:
                metric(outputs, target)

    val_loss /= len(dataloader)
    return val_loss, metric


def fit(train_dataloader, dev_dataloader, model, loss_fn, optimizer, scheduler, early_stopping,
        epochs, device, metric=None, log_interval=1000, ordered_dataloader=None):
    model = model.to(device)
    for epoch in range(epochs):
        model.start_train()
        train_loss, metric = train_epoch(train_dataloader, model, loss_fn, optimizer,
                                         scheduler, device, metric, log_interval)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, epochs, train_loss)
        if metric is not None:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        logger.info(message)

        if ordered_dataloader is not None:
            logger.info("Update datastore.")
            batch_size = 0
            logger.info('Generate Datastore.')
            model.knn_store.reset_dstore()
            model.eval()
            for i, (data, target) in enumerate(tqdm(ordered_dataloader)):
                if batch_size == 0: batch_size = target.size()[0]
                with torch.no_grad():
                    data = tuple(d.to(device) for d in data)
                    target = target.unsqueeze(-1)
                    _, output = model.pretrain_model(*data)
                    model.knn_store.write_dstore(output, target, i * batch_size, batch_size)
            model.knn_store.add_index()

        model.start_test()
        val_loss, metric = test_epoch(dev_dataloader, model, loss_fn, device, metric)

        message = 'Epoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, epochs, val_loss)
        if metric is not None:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        logger.info(message)

        if early_stopping.compare_loss:
            early_stopping(val_loss, model)
        else:
            early_stopping(metric.value(), model)

        if early_stopping.early_stop:
            logger.info("Early stopping!")
            break


def no_args_train(dev_dataloader, test_dataloader, model, device, candidate_k, candidate_tmp, candidate_other=None):
    model.to(device)
    model.eval()
    logger.info('Find best hyperparameters.')
    dev_acc = [[[0] for _ in range(len(candidate_tmp))]
               for _ in range(len(candidate_k))] if candidate_other is None else \
        [[[0 for _ in range(len(candidate_other))]
          for _ in range(len(candidate_tmp))] for _ in range(len(candidate_k))]

    cnt = 0
    for data, target in tqdm(dev_dataloader):
        cnt += target.size()[0]
        data = tuple(d.to(device) for d in data)
        target = target.to(device)
        for i, k in enumerate(candidate_k):
            for j, temperature in enumerate(candidate_tmp):
                loop_list = [0] if candidate_other is None else candidate_other
                for z, param in enumerate(loop_list):
                    if candidate_other is None:
                        input = data + (k, temperature)
                    else:
                        input = data + (k, temperature, param)
                    prob = model(*input)
                    dev_acc[i][j][z] += (torch.argmax(prob, dim=1) == target).sum().item()

    best_dev_acc = 0
    best_k = 0
    best_temp = 0
    best_param = 0
    for i, tmp_i in enumerate(dev_acc):
        for j, tmp_j in enumerate(tmp_i):
            for z, _acc in enumerate(tmp_j):
                _acc /= cnt
                if _acc > best_dev_acc:
                    best_dev_acc = _acc
                    best_k = candidate_k[i]
                    best_temp = candidate_tmp[j]
                    if candidate_other is not None:
                        best_param = candidate_other[z]
                if candidate_other is not None:
                    logger.info(
                        f"Params: {candidate_k[i]}\t{candidate_tmp[j]}\t{candidate_other[z]}\tdev_acc: {_acc:.5f}")
                else:
                    logger.info(f"Params: {candidate_k[i]}\t{candidate_tmp[j]}\tdev_acc: {_acc:.5f}")

    if candidate_other is None:
        logger.info(f'*Max dev accuracy: {best_dev_acc} (k={best_k}, T={best_temp})')
    else:
        logger.info(f'*Max dev accuracy: {best_dev_acc} (k={best_k}, T={best_temp}, w/th={best_param})')

    # best_k = 4
    # best_temp = 10
    # best_param = 0.9

    logger.info('Start testing.')
    cnt = 0
    test_acc = 0
    for data, target in test_dataloader:
        cnt += target.size()[0]
        data = tuple(d.to(device) for d in data)
        target = target.to(device)
        data = data + (best_k, best_temp) if candidate_other is None else data + (best_k, best_temp, best_param)
        prob = model(*data)
        test_acc += (torch.argmax(prob, dim=1) == target).sum().item()
    logger.info(f"*Test accuracy: {test_acc / cnt:.5f}")
