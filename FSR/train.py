import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.resnet_fsr import ResNet18_FSR
from models.vgg_fsr import vgg16_FSR
from models.wideresnet34_fsr import WideResNet34_FSR
from models.SeparateClassifier import SeparateClassifier

from attacks.pgd import PGD

from tqdm.auto import tqdm
import argparse
import os

from datasets import available_datasets
import logging



def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='FSR Training')
parser.add_argument('--save_name', type=str, help='specify checkpoint save name')
parser.add_argument('--lam_sep', type=float, default=1.0, help='weight for separation loss')
parser.add_argument('--lam_rec', type=float, default=1.0, help='weight for recalibration loss')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate for classifier')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
parser.add_argument('--dataset', type=str, default='cifar10', help='target dataset')
parser.add_argument('--model', type=str, default='resnet18', help='model name')
parser.add_argument('--eps', type=float, default=8., help='perturbation constraint epsilon')
parser.add_argument('--alpha', type=float, default=0.25, help='step size alpha')
parser.add_argument('--tau', type=float, default=0.1, help='tau for Gumbel softmax')
parser.add_argument('--device', type=int, help='device id')

### Arguments for transfer learning
parser.add_argument('--transfer_learning', action='store_true', help='train the last layer of FSR only')
parser.add_argument('--backbone_weights', type=str, help='path to backbone weights')
parser.add_argument('--fsr_weights', type=str, help='path to FSR weights')
args = parser.parse_args()

# Write logs to file
if not os.path.exists('./logs/train'):
    os.makedirs('./logs/train')
log_file = './logs/train/{}.txt'.format(args.save_name)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

logger.info("Training Configuration:")
for arg, value in vars(args).items():
    logger.info(f'{arg}: {value}')


device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
start_epoch = 1

dataset = available_datasets[args.dataset](args)
(num_classes, image_size,
 trainloader, testloader, trainset, testset) = dataset.get_dataset()

models = {
    'resnet18': ResNet18_FSR,
    'vgg16': vgg16_FSR,
    'wideresnet34': WideResNet34_FSR,
}

model_name = args.model
net = models[model_name]

# Transfer learning
if args.transfer_learning:
    classifier = SeparateClassifier(args=args, num_classes=num_classes, image_size=image_size, net=net, device=device)
    classifier.load(args)
    classifier.set_requires_grad([classifier.net.tranfer_trainable_layer], True)
    net = classifier.net

net = net.to(device)
cudnn.benchmark = True


criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)



def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)

    return adv_label


attack = PGD(net, args.eps/255.0, args.alpha * (args.eps/255.0), min_val=0, max_val=1, max_iters=10, _type='linf')


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    adv_cls_losses = 0
    sep_losses = 0
    rec_losses = 0
    adv_correct = 0
    total = 0

    adjust_learning_rate(optimizer, epoch)

    with tqdm(total=(len(trainset) - len(trainset) % args.bs)) as _tqdm:
        _tqdm.set_description('{} (Train) Epoch: {}/{}'.format(args.save_name, epoch, args.epoch))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            net.eval()
            adv_inputs = attack.perturb(inputs, targets, True)
            net.train()

            adv_outputs, adv_r_outputs, adv_nr_outputs, adv_rec_outputs = net(adv_inputs)
            adv_labels = get_pred(adv_outputs, targets)

            adv_cls_loss = criterion(adv_outputs, targets)
            
            r_loss = torch.tensor(0.).to(device)
            if not len(adv_r_outputs) == 0:
                for r_out in adv_r_outputs:
                    r_loss += args.lam_sep * criterion(r_out, targets)
                r_loss /= len(adv_r_outputs)

            nr_loss = torch.tensor(0.).to(device)
            if not len(adv_nr_outputs) == 0:
                for nr_out in adv_nr_outputs:
                    nr_loss += args.lam_sep * criterion(nr_out, adv_labels)
                nr_loss /= len(adv_nr_outputs)
            sep_loss = r_loss + nr_loss

            rec_loss = torch.tensor(0.).to(device)
            if not len(adv_rec_outputs) == 0:
                for rec_out in adv_rec_outputs:
                    rec_loss += args.lam_rec * criterion(rec_out, targets)
                rec_loss /= len(adv_rec_outputs)

            loss = adv_cls_loss + sep_loss + rec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            adv_cls_losses += adv_cls_loss.item()
            sep_losses += sep_loss.item()
            rec_losses += rec_loss.item()
            _, adv_predicted = adv_outputs.max(1)
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()

            _tqdm.set_postfix(
                Adv_Loss='{:.3f}'.format(adv_cls_losses / (batch_idx + 1)),
                Sep_Loss='{:.3f}'.format(sep_losses / (batch_idx + 1)),
                Rec_Loss='{:.3f}'.format(rec_losses / (batch_idx + 1)),
                Adv_Acc='{:.3f}%'.format(100. * adv_correct / total),
            )
            _tqdm.update(inputs.shape[0])
            
    log_msg = ('Epoch: %d, Adv_Loss: %.3f, Sep_Loss: %.3f, Rec_Loss: %.3f, '
                'Adv_Acc: %.3f%%' % (epoch, adv_cls_losses / len(trainloader),
                                    sep_losses / len(trainloader),
                                    rec_losses / len(trainloader),
                                    100. * adv_correct / total))
    logging.info(log_msg)


def test(epoch):
    net.eval()
    ori_test_loss = 0
    adv_test_loss = 0
    ori_correct = 0
    adv_correct = 0
    total = 0
    with tqdm(total=(len(testset) - len(testset) % args.bs), dynamic_ncols=True) as _tqdm:
        _tqdm.set_description('{} (Test) Epoch: {}/{}'.format(args.save_name, epoch, args.epoch))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_inputs = attack.perturb(inputs, targets, False)
            net.eval()

            ori_outputs, _, _, _ = net(inputs, is_eval=True)
            adv_outputs, _, _, _ = net(adv_inputs, is_eval=True)

            ori_loss = criterion(ori_outputs, targets)
            ori_test_loss += ori_loss.item()
            _, ori_predicted = ori_outputs.max(1)
            ori_correct += ori_predicted.eq(targets).sum().item()

            adv_loss = criterion(adv_outputs, targets)
            adv_test_loss += adv_loss.item()
            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(targets).sum().item()

            total += targets.size(0)

            _tqdm.set_postfix(
                Ori_Loss='{:.3f}'.format(ori_test_loss/(batch_idx+1)),
                Ori_Acc='{:.3f}%'.format(100.*ori_correct/total),
                Adv_Loss='{:.3f}'.format(adv_test_loss/(batch_idx+1)),
                Adv_Acc='{:.3f}%'.format(100.*adv_correct/total),
            )
            _tqdm.update(inputs.shape[0])

    log_msg = ('Epoch: %d, Ori_Loss: %.3f, Ori_Acc: %.3f%%, '
                'Adv_Loss: %.3f, Adv_Acc: %.3f%%' % (epoch,
                                                    ori_test_loss/len(testloader),
                                                    100.*ori_correct/total,
                                                    adv_test_loss/len(testloader),
                                                    100.*adv_correct/total))
    logging.info(log_msg)
    
    if not os.path.exists('./weights/{}/{}/'.format(args.dataset, args.model)):
        os.makedirs('./weights/{}/{}/'.format(args.dataset, args.model))
    torch.save(net.state_dict(), './weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.save_name))


for epoch in range(start_epoch, args.epoch + 1):
    train(epoch)
    test(epoch)
