# Backbone: MNIST
# FSR: CIFAR10, CIFAR100, MNIST, SVHN, TinyImageNet

# MNIST -> CIFAR10
python train.py \
--save_name trsf_bbMNIST_fsrCifar10_resnet18 --dataset mnist --model resnet18 --device 2 --epoch 20 \
--transfer_learning --backbone_weights weights/mnist/resnet18/mnist_resnet18_fsrTest.pth --fsr_weights weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth

# MNIST -> CIFAR100
python train.py \
--save_name trsf_bbMNIST_fsrCifar100_resnet18 --dataset mnist --model resnet18 --device 2 --epoch 20 \
--transfer_learning --backbone_weights weights/mnist/resnet18/mnist_resnet18_fsrTest.pth --fsr_weights weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth

# MNIST -> MNIST
python train.py \
--save_name trsf_bbMNIST_fsrMNIST_resnet18 --dataset mnist --model resnet18 --device 2 --epoch 20 \
--transfer_learning --backbone_weights weights/mnist/resnet18/mnist_resnet18_fsrTest.pth --fsr_weights weights/mnist/resnet18/mnist_resnet18_fsrTest.pth

# MNIST -> SVHN
python train.py \
--save_name trsf_bbMNIST_fsrSVHN_resnet18 --dataset mnist --model resnet18 --device 2 --epoch 20 \
--transfer_learning --backbone_weights weights/mnist/resnet18/mnist_resnet18_fsrTest.pth --fsr_weights weights/svhn/resnet18/svhn_resnet18_fsrTest.pth

# MNIST -> TinyImageNet
python train.py \
--save_name trsf_bbMNIST_fsrTinyimg_resnet18 --dataset mnist --model resnet18 --device 2 --epoch 20 \
--transfer_learning --backbone_weights weights/mnist/resnet18/mnist_resnet18_fsrTest.pth --fsr_weights weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth