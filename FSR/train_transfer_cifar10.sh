# Backbone: CIFAR10
# FSR: CIFAR10, CIFAR100, MNIST, SVHN, TinyImageNet

# CIFAR10 -> CIFAR10
python train.py \
--save_name trsf_bbCifar10_fsrCifar10_resnet18 --dataset cifar10 --model resnet18 --device 0 --epoch 100 \
--transfer_learning --backbone_weights weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth --fsr_weights weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth

# CIFAR10 -> CIFAR100
python train.py \
--save_name trsf_bbCifar10_fsrCifar100_resnet18 --dataset cifar10 --model resnet18 --device 0 --epoch 100 \
--transfer_learning --backbone_weights weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth --fsr_weights weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth

# CIFAR10 -> MNIST
python train.py \
--save_name trsf_bbCifar10_fsrMNIST_resnet18 --dataset cifar10 --model resnet18 --device 0 --epoch 100 \
--transfer_learning --backbone_weights weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth --fsr_weights weights/mnist/resnet18/mnist_resnet18_fsrTest.pth

# CIFAR10 -> SVHN
python train.py \
--save_name trsf_bbCifar10_fsrSVHN_resnet18 --dataset cifar10 --model resnet18 --device 0 --epoch 100 \
--transfer_learning --backbone_weights weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth --fsr_weights weights/svhn/resnet18/svhn_resnet18_fsrTest.pth

# CIFAR10 -> TinyImageNet
python train.py \
--save_name trsf_bbCifar10_fsrTinyimg_resnet18 --dataset cifar10 --model resnet18 --device 0 --epoch 100 \
--transfer_learning --backbone_weights weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth --fsr_weights weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth