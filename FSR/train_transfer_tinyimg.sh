# Backbone: TinyImageNet
# FSR: CIFAR10, CIFAR100, MNIST, SVHN, TinyImageNet

# TinyImageNet -> CIFAR10
python train.py \
--save_name trsf_bbTinyimg_fsrCifar10_resnet18 --dataset tinyimg --model resnet18 --device 4 --epoch 100 \
--transfer_learning --backbone_weights weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth --fsr_weights weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth

# TinyImageNet -> CIFAR100
python train.py \
--save_name trsf_bbTinyimg_fsrCifar100_resnet18 --dataset tinyimg --model resnet18 --device 4 --epoch 100 \
--transfer_learning --backbone_weights weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth --fsr_weights weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth

# TinyImageNet -> MNIST
python train.py \
--save_name trsf_bbTinyimg_fsrMNIST_resnet18 --dataset tinyimg --model resnet18 --device 4 --epoch 100 \
--transfer_learning --backbone_weights weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth --fsr_weights weights/mnist/resnet18/mnist_resnet18_fsrTest.pth

# TinyImageNet -> SVHN
python train.py \
--save_name trsf_bbTinyimg_fsrSVHN_resnet18 --dataset tinyimg --model resnet18 --device 4 --epoch 100 \
--transfer_learning --backbone_weights weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth --fsr_weights weights/svhn/resnet18/svhn_resnet18_fsrTest.pth

# TinyImageNet -> TinyImageNet
python train.py \
--save_name trsf_bbTinyimg_fsrTinyimg_resnet18 --dataset tinyimg --model resnet18 --device 4 --epoch 100 \
--transfer_learning --backbone_weights weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth --fsr_weights weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth