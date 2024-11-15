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

import torchvision
import torchvision.transforms as transforms

from models.BaseModel import BaseModelDNN
from datasets import available_datasets
import logging


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--load_name', type=str, help='specify checkpoint load name')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--tau', default=0.1, type=float)
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--backbone_weights', type=str, help='path to backbone weights')
parser.add_argument('--fsr_weights', type=str, help='path to FSR weights')
args = parser.parse_args()

# Write logs to file
backbone_name = args.backbone_weights.split('/')[-1].split('_')[0]
fsr_name = args.fsr_weights.split('/')[-1].split('_')[0]
if not os.path.exists('./logs/test'):
    os.makedirs('./logs/test')
log_file = f'./logs/test/bb_{backbone_name}_fsr_{fsr_name}.txt'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

logger.info("Test Configuration:")
for arg, value in vars(args).items():
    logger.info(f'{arg}: {value}')
    

device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'


if args.model == 'resnet18':
    from models.resnet_fsr import ResNet18_FSR
    net = ResNet18_FSR

elif args.model == 'vgg16':
    from models.vgg_fsr import vgg16_FSR
    net = vgg16_FSR

elif args.model == 'wideresnet34':
    from models.wideresnet34_fsr import WideResNet34_FSR
    net = WideResNet34_FSR

dataset = available_datasets[args.dataset](args)
(num_classes, image_size,
    trainloader, testloader, trainset, testset) = dataset.get_dataset()


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
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits_final, target):
        loss = self._cw_loss(logits_final, target, num_classes=self.num_classes)

        return loss

    def _cw_loss(self, output, target, confidence=50, num_classes=10):
        target = target.data
        target_onehot = torch.zeros(target.size() + (num_classes,))
        target_onehot = target_onehot.to(device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = Variable(target_onehot, requires_grad=False)
        real = (target_var * output).sum(1)
        other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
        loss = torch.sum(loss)
        return loss


class Classifier(BaseModelDNN):
    def __init__(self) -> None:
        super(BaseModelDNN).__init__()
        self.net = net(tau=args.tau, num_classes=num_classes, image_size=image_size).to(device)
        self.set_requires_grad([self.net], False)

    def load(self, args):
        checkpoint = torch.load('./weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.load_name), map_location=device)
        self.net.load_state_dict(checkpoint)
    
    def predict(self, x, is_eval=True):
        return self.net(x, is_eval=is_eval)

class TransferClassifier(BaseModelDNN):
    # Model for transfer learning, the backbone & FSR module might be trained on different datasets
    def __init__(self):
        super(BaseModelDNN).__init__()
        self.net = net(tau=args.tau, num_classes=num_classes, image_size=image_size).to(device)
        self.set_requires_grad([self.net], False)
        
    def load(self, args):
        print(f'Loading model with backbone: {args.backbone_weights} and FSR: {args.fsr_weights}')
        backbone_checkpoint = torch.load(args.backbone_weights, map_location=device)
        self.net.load_state_dict(backbone_checkpoint)
        
        # Replace the FSR module
        fsr_checkpoint = torch.load(args.fsr_weights, map_location=device)
        fsr_state_dict = {
            k.replace('fsr.', ''): v
            for k, v in fsr_checkpoint.items()
            if k.startswith('fsr.')
        }

        self.net.fsr.load_state_dict(fsr_state_dict)

    def predict(self, x, is_eval=True):
        return self.net(x, is_eval=is_eval)
        


def main():
    # model = Classifier()
    model = TransferClassifier()
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
