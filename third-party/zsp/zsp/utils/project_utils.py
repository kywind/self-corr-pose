import os
import torch
import random
import numpy as np
import inspect

from typing import Optional
from torch.optim.optimizer import Optimizer

from datetime import datetime

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def strip_state_dict(state_dict, strip_key='module.'):

    """
    Strip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict


def transform_moco_state_dict(obj, num_classes):

    """
    :param obj: Moco State Dict
    :param args: argsparse object with training classes
    :return: State dict compatable with standard resnet architecture
    """

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder_q."):
            continue
        old_k = k
        k = k.replace("module.encoder_q.", "")

        if k.startswith("fc.2"):
            continue

        if k.startswith("fc.0"):
            k = k.replace("0.", "")
            if "weight" in k:
                v = torch.randn((num_classes, v.size(1)))
            elif "bias" in k:
                v = torch.randn((num_classes,))

        newmodel[k] = v

    return newmodel



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ClassificationPredSaver(object):

    def __init__(self, length, save_path=None):

        if save_path is not None:

            # Remove filetype from save_path
            save_path = save_path.split('.')[0]
            self.save_path = save_path

        self.length = length

        self.all_preds = None
        self.all_labels = None

        self.running_start_idx = 0

    def update(self, preds, labels=None):

        # Expect preds in shape B x C

        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()

        b, c = preds.shape

        if self.all_preds is None:
            self.all_preds = np.zeros((self.length, c))

        self.all_preds[self.running_start_idx: self.running_start_idx + b] = preds

        if labels is not None:
            if torch.is_tensor(labels):
                labels = labels.detach().cpu().numpy()

            if self.all_labels is None:
                self.all_labels = np.zeros((self.length,))

            self.all_labels[self.running_start_idx: self.running_start_idx + b] = labels

        # Maintain running index on dataset being evaluated
        self.running_start_idx += b

    def save(self):

        # Softmax over preds
        preds = torch.from_numpy(self.all_preds)
        preds = torch.nn.Softmax(dim=-1)(preds)
        self.all_preds = preds.numpy()

        pred_path = self.save_path + '.pth'
        print(f'Saving all predictions to {pred_path}')

        torch.save(self.all_preds, pred_path)

        if self.all_labels is not None:

            # Evaluate
            self.evaluate()
            torch.save(self.all_labels, self.save_path + '_labels.pth')

    def evaluate(self):

        topk = [1, 5, 10]
        topk = [k for k in topk if k < self.all_preds.shape[-1]]
        acc = accuracy(torch.from_numpy(self.all_preds), torch.from_numpy(self.all_labels), topk=topk)

        for k, a in zip(topk, acc):
            print(f'Top{k} Acc: {a.item()}')


def get_acc_auroc_curves(logdir):

    """
    :param logdir: Path to logs: E.g '/work/me/logs/'
    :return:
    """

    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    # Only gets scalars
    log_info = {}
    for tag in event_acc.Tags()['scalars']:

        log_info[tag] = np.array([[x.step, x.value] for x in event_acc.scalars._buckets[tag].items])

    return log_info


def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()

class IndicatePlateau(object):

    def __init__(self, threshold=5e-4, patience_epochs=5, mode='min', threshold_mode='rel'):

        self.patience = patience_epochs
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)

        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            print('Tracked metric has plateaud')
            self._reset()
            return True
        else:
            return False

    def is_better(self, a, best):

        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):

        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = -float('inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


def denorm_image(image, mean, std):
    image = image * std[:, None, None]
    image = image + mean[:, None, None]
    return image


def square_bbox(bbox):
    '''
    Converts a bbox to have a square shape by increasing size along non-max dimension.

    https://github.com/akanazawa/cmr
    '''
    sq_bbox = [int(round(coord)) for coord in bbox]
    bwidth = sq_bbox[2] - sq_bbox[0] + 1
    bheight = sq_bbox[3] - sq_bbox[1] + 1
    maxdim = float(max(bwidth, bheight))

    dw_b_2 = int(round((maxdim-bwidth)/2.0))
    dh_b_2 = int(round((maxdim-bheight)/2.0))

    sq_bbox[0] -= dw_b_2
    sq_bbox[1] -= dh_b_2
    sq_bbox[2] = sq_bbox[0] + maxdim - 1
    sq_bbox[3] = sq_bbox[1] + maxdim - 1

    return sq_bbox


if __name__ == '__main__':

    x = IndicatePlateau(threshold=0.0899)
    eps = np.arange(0, 2000, 1)
    y = np.exp(-0.01 * eps)

    print(y)
    for i, y_ in enumerate(y):

        z = x.step(y_)
        if z:
            print(f'Plateaud at epoch {i} with val {y_}')

def get_results_length(results_file):
    if os.path.exists(results_file):
        lines = list(open(results_file, 'r'))
        if 'THIS CATEGORY BROKE' in lines:
            lines.remove('THIS CATEGORY BROKE')
        return len(lines)
    else:
        return 0
