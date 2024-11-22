'''
Some parts of the code are modified from:
CAS : https://github.com/bymavis/CAS_ICLR2021
CIFS : https://github.com/HanshuYAN/CIFS
'''


import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.WholeClassifier import WholeClassifier
from models.SeparateClassifier import SeparateClassifier
from models.resnet_fsr import ResNet18_FSR
from models.vgg_fsr import vgg16_FSR
from models.wideresnet34_fsr import WideResNet34_FSR

from datasets import available_datasets
import logging


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_arguments():
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--load_name', type=str, help='specify checkpoint load name')
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--backbone_weights', type=str, help='path to backbone weights')
    parser.add_argument('--fsr_weights', type=str, help='path to FSR weights')
    # Transfer learning arguments
    parser.add_argument('--transfer_learning', action='store_true', help='evaluate the trained transfer learning model')
    parser.add_argument('--transfer_weights', type=str, help='path to transfer learning model weights')
    args = parser.parse_args()
    
    if not args.transfer_learning:
        assert args.backbone_weights is not None, 'Please provide the backbone weights for non-transfer learning'
        assert args.fsr_weights is not None, 'Please provide the FSR weights for non-transfer learning'
    else:
        assert args.transfer_weights is not None, 'Please provide the transfer model weights for transfer learning'
    
    return args

def set_logger(args):
    if not os.path.exists('./logs/test'):
        os.makedirs('./logs/test')
    if args.transfer_learning:
        log_file = f'./logs/test/trsf_{args.transfer_weights.split("/")[-1].split("_")[0]}.txt'
    else:
        backbone_name = args.backbone_weights.split('/')[-1].split('_')[0]
        fsr_name = args.fsr_weights.split('/')[-1].split('_')[0]
        log_file = f'./logs/test/bb_{backbone_name}_fsr_{fsr_name}.txt'
    
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.info("Test Configuration:")
    for arg, value in vars(args).items():
        logger.info(f'{arg}: {value}')
        
    return logger, log_file

def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)

    return adv_label


class CE_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits_final, target):
        loss = F.cross_entropy(logits_final, target)

        return loss


class CW_loss(nn.Module):
    def __init__(self, num_classes=10, device='cuda:0') -> None:
        super().__init__()
        self.num_classes = num_classes
        self.device = device

    def forward(self, logits_final, target):
        loss = self._cw_loss(logits_final, target, num_classes=self.num_classes)

        return loss

    def _cw_loss(self, output, target, confidence=50, num_classes=10):
        target = target.data
        target_onehot = torch.zeros(target.size() + (num_classes,))
        target_onehot = target_onehot.to(self.device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = Variable(target_onehot, requires_grad=False)
        real = (target_var * output).sum(1)
        other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
        loss = torch.sum(loss)
        return loss


# class Classifier(BaseModelDNN):
#     def __init__(self) -> None:
#         super(BaseModelDNN).__init__()
#         self.net = net(tau=args.tau, num_classes=num_classes, image_size=image_size).to(device)
#         self.set_requires_grad([self.net], False)

#     def load(self, args):
#         checkpoint = torch.load('./weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.load_name), map_location=device)
#         self.net.load_state_dict(checkpoint)
    
#     def predict(self, x, is_eval=True):
#         return self.net(x, is_eval=is_eval)

def main():
    args = get_arguments()
    logger, log_file = set_logger(args)
    
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    nets = {
        'resnet18': ResNet18_FSR,
        'vgg16': vgg16_FSR,
        'wideresnet34': WideResNet34_FSR,
    }
    net = nets[args.model]
    dataset = available_datasets[args.dataset](args)
    (num_classes, image_size,
        trainloader, testloader, trainset, testset) = dataset.get_dataset()

    if args.transfer_learning:
        model = WholeClassifier(args=args, num_classes=num_classes, image_size=image_size, net=net, device=device)
    else:
        model = SeparateClassifier(args=args, num_classes=num_classes, image_size=image_size, net=net, device=device)
    model.load(args)
    model.net.eval()

    from advertorch_fsr.attacks import FGSM, LinfPGDAttack

    lst_attack = [
        (FGSM, dict(
            loss_fn=CE_loss(),
            eps=8 / 255,
            clip_min=0.0, clip_max=1.0, targeted=False), 'FGSM'),
        (LinfPGDAttack, dict(
            loss_fn=CE_loss(),
            eps=8 / 255, nb_iter=20, eps_iter=0.1 * (8 / 255), rand_init=False,
            clip_min=0.0, clip_max=1.0, targeted=False), 'PGD-20'),
        (LinfPGDAttack, dict(
            loss_fn=CE_loss(),
            eps=8 / 255, nb_iter=100, eps_iter=0.1 * (8 / 255), rand_init=False,
            clip_min=0.0, clip_max=1.0, targeted=False), 'PGD-100'),
        (LinfPGDAttack, dict(
            loss_fn=CW_loss(num_classes=num_classes),
            eps=8 / 255, nb_iter=30, eps_iter=0.1 * (8 / 255), rand_init=False,
            clip_min=0.0, clip_max=1.0, targeted=False), 'C&W'),
    ]
    attack_results = []
    for attack_class, attack_kwargs, name in lst_attack:
        from metric.classification import defense_success_rate
        print(f'Processing {name} attack...')
        message, defense_success, natural_success = defense_success_rate(model.predict,
                                                                         testloader, attack_class,
                                                                         attack_kwargs, device=device)

        message = name + ': ' + message
        logger.info(message)
        attack_results.append(defense_success)
    attack_results.append(natural_success)
    attack_results = torch.cat(attack_results, 1)
    attack_results = attack_results.sum(1)
    attack_results[attack_results < len(lst_attack) + 1] = 0. # The defense is considered successful only if all attacks failed
    logger.info('Ensemble : {:.2f}%'.format(100. * attack_results.count_nonzero() / len(testset))
                + ' ({}/{})'.format(attack_results.count_nonzero(), len(testset)))
    
    print(f'Results saved to {log_file}.')

if __name__ == '__main__':
    main()
