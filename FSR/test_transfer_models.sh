# Source data: cifar10
# python test.py --load_name cifar10_resnet18 --dataset cifar10 --model resnet18 \
# --transfer_learning --device 7 \
# --backbone_weights ./weights/cifar10/resnet18/trsf_bbCifar10_fsrCifar10_resnet18.pth \
# --fsr_weights ./weights/cifar10/resnet18/trsf_bbCifar10_fsrCifar10_resnet18.pth
python test.py --load_name cifar10_resnet18 --dataset cifar10 --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/cifar10/resnet18/trsf_bbCifar10_fsrCifar100_resnet18.pth \
--fsr_weights ./weights/cifar10/resnet18/trsf_bbCifar10_fsrCifar100_resnet18.pth
python test.py --load_name cifar10_resnet18 --dataset cifar10 --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/cifar10/resnet18/trsf_bbCifar10_fsrMNIST_resnet18.pth \
--fsr_weights ./weights/cifar10/resnet18/trsf_bbCifar10_fsrMNIST_resnet18.pth
python test.py --load_name cifar10_resnet18 --dataset cifar10 --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/cifar10/resnet18/trsf_bbCifar10_fsrSVHN_resnet18.pth \
--fsr_weights ./weights/cifar10/resnet18/trsf_bbCifar10_fsrSVHN_resnet18.pth

# Source data: cifar100
python test.py --load_name cifar100_resnet18 --dataset cifar100 --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/cifar100/resnet18/trsf_bbCifar100_fsrCifar10_resnet18.pth \
--fsr_weights ./weights/cifar100/resnet18/trsf_bbCifar100_fsrCifar10_resnet18.pth
python test.py --load_name cifar100_resnet18 --dataset cifar100 --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/cifar100/resnet18/trsf_bbCifar100_fsrCifar100_resnet18.pth \
--fsr_weights ./weights/cifar100/resnet18/trsf_bbCifar100_fsrCifar100_resnet18.pth
python test.py --load_name cifar100_resnet18 --dataset cifar100 --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/cifar100/resnet18/trsf_bbCifar100_fsrMNIST_resnet18.pth \
--fsr_weights ./weights/cifar100/resnet18/trsf_bbCifar100_fsrMNIST_resnet18.pth
python test.py --load_name cifar100_resnet18 --dataset cifar100 --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/cifar100/resnet18/trsf_bbCifar100_fsrSVHN_resnet18.pth \
--fsr_weights ./weights/cifar100/resnet18/trsf_bbCifar100_fsrSVHN_resnet18.pth

# Source data: mnist
python test.py --load_name mnist_resnet18 --dataset mnist --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/mnist/resnet18/trsf_bbMNIST_fsrCifar10_resnet18.pth \
--fsr_weights ./weights/mnist/resnet18/trsf_bbMNIST_fsrCifar10_resnet18.pth
python test.py --load_name mnist_resnet18 --dataset mnist --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/mnist/resnet18/trsf_bbMNIST_fsrCifar100_resnet18.pth \
--fsr_weights ./weights/mnist/resnet18/trsf_bbMNIST_fsrCifar100_resnet18.pth
python test.py --load_name mnist_resnet18 --dataset mnist --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/mnist/resnet18/trsf_bbMNIST_fsrMNIST_resnet18.pth \
--fsr_weights ./weights/mnist/resnet18/trsf_bbMNIST_fsrMNIST_resnet18.pth
python test.py --load_name mnist_resnet18 --dataset mnist --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/mnist/resnet18/trsf_bbMNIST_fsrSVHN_resnet18.pth \
--fsr_weights ./weights/mnist/resnet18/trsf_bbMNIST_fsrSVHN_resnet18.pth

# Source data: svhn
python test.py --load_name svhn_resnet18 --dataset svhn --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/svhn/resnet18/trsf_bbSVHN_fsrCifar10_resnet18.pth \
--fsr_weights ./weights/svhn/resnet18/trsf_bbSVHN_fsrCifar10_resnet18.pth
python test.py --load_name svhn_resnet18 --dataset svhn --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/svhn/resnet18/trsf_bbSVHN_fsrCifar100_resnet18.pth \
--fsr_weights ./weights/svhn/resnet18/trsf_bbSVHN_fsrCifar100_resnet18.pth
python test.py --load_name svhn_resnet18 --dataset svhn --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/svhn/resnet18/trsf_bbSVHN_fsrMNIST_resnet18.pth \
--fsr_weights ./weights/svhn/resnet18/trsf_bbSVHN_fsrMNIST_resnet18.pth
python test.py --load_name svhn_resnet18 --dataset svhn --model resnet18 \
--transfer_learning --device 7 \
--backbone_weights ./weights/svhn/resnet18/trsf_bbSVHN_fsrSVHN_resnet18.pth \
--fsr_weights ./weights/svhn/resnet18/trsf_bbSVHN_fsrSVHN_resnet18.pth
