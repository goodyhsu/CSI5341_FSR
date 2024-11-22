# Backbone: SVHN
# FSR: CIFAR10, CIFAR100, MNIST, SVHN, TinyImageNet

# SVHN -> CIFAR10
python train.py \
--save_name trsf_bbSVHN_fsrCifar10_resnet18 --dataset svhn --model resnet18 --device 3 --epoch 100 \
--transfer_learning --backbone_weights weights/svhn/resnet18/svhn_resnet18_fsrTest.pth --fsr_weights weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth

# SVHN -> CIFAR100
python train.py \
--save_name trsf_bbSVHN_fsrCifar100_resnet18 --dataset svhn --model resnet18 --device 3 --epoch 100 \
--transfer_learning --backbone_weights weights/svhn/resnet18/svhn_resnet18_fsrTest.pth --fsr_weights weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth

# SVHN -> MNIST
python train.py \
--save_name trsf_bbSVHN_fsrMNIST_resnet18 --dataset svhn --model resnet18 --device 3 --epoch 100 \
--transfer_learning --backbone_weights weights/svhn/resnet18/svhn_resnet18_fsrTest.pth --fsr_weights weights/mnist/resnet18/mnist_resnet18_fsrTest.pth

# SVHN -> SVHN
python train.py \
--save_name trsf_bbSVHN_fsrSVHN_resnet18 --dataset svhn --model resnet18 --device 3 --epoch 100 \
--transfer_learning --backbone_weights weights/svhn/resnet18/svhn_resnet18_fsrTest.pth --fsr_weights weights/svhn/resnet18/svhn_resnet18_fsrTest.pth

# SVHN -> TinyImageNet
python train.py \
--save_name trsf_bbSVHN_fsrTinyimg_resnet18 --dataset svhn --model resnet18 --device 3 --epoch 100 \
--transfer_learning --backbone_weights weights/svhn/resnet18/svhn_resnet18_fsrTest.pth --fsr_weights weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth