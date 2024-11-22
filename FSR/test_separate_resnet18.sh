# Source data: cifar10
python test.py --load_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 7 \
--backbone_weights ./weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth \
--fsr_weights ./weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth
python test.py --load_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 7 \
--backbone_weights ./weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth \
--fsr_weights ./weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth
python test.py --load_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 7 \
--backbone_weights ./weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth \
--fsr_weights ./weights/mnist/resnet18/mnist_resnet18_fsrTest.pth
python test.py --load_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 7 \
--backbone_weights ./weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth \
--fsr_weights ./weights/svhn/resnet18/svhn_resnet18_fsrTest.pth

# Source data: cifar100
python test.py --load_name cifar100_resnet18 --dataset cifar100 --model resnet18 --device 7 \
--backbone_weights ./weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth \
--fsr_weights ./weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth
python test.py --load_name cifar100_resnet18 --dataset cifar100 --model resnet18 --device 7 \
--backbone_weights ./weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth \
--fsr_weights ./weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth
python test.py --load_name cifar100_resnet18 --dataset cifar100 --model resnet18 --device 7 \
--backbone_weights ./weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth \
--fsr_weights ./weights/mnist/resnet18/mnist_resnet18_fsrTest.pth
python test.py --load_name cifar100_resnet18 --dataset cifar100 --model resnet18 --device 7 \
--backbone_weights ./weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth \
--fsr_weights ./weights/svhn/resnet18/svhn_resnet18_fsrTest.pth

# Source data: mnist
python test.py --load_name mnist_resnet18 --dataset mnist --model resnet18 --device 7 \
--backbone_weights ./weights/mnist/resnet18/mnist_resnet18_fsrTest.pth \
--fsr_weights ./weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth
python test.py --load_name mnist_resnet18 --dataset mnist --model resnet18 --device 7 \
--backbone_weights ./weights/mnist/resnet18/mnist_resnet18_fsrTest.pth \
--fsr_weights ./weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth
python test.py --load_name mnist_resnet18 --dataset mnist --model resnet18 --device 7 \
--backbone_weights ./weights/mnist/resnet18/mnist_resnet18_fsrTest.pth \
--fsr_weights ./weights/mnist/resnet18/mnist_resnet18_fsrTest.pth
python test.py --load_name mnist_resnet18 --dataset mnist --model resnet18 --device 7 \
--backbone_weights ./weights/mnist/resnet18/mnist_resnet18_fsrTest.pth \
--fsr_weights ./weights/svhn/resnet18/svhn_resnet18_fsrTest.pth

# Source data: svhn
python test.py --load_name svhn_resnet18 --dataset svhn --model resnet18 --device 7 \
--backbone_weights ./weights/svhn/resnet18/svhn_resnet18_fsrTest.pth \
--fsr_weights ./weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth
python test.py --load_name svhn_resnet18 --dataset svhn --model resnet18 --device 7 \
--backbone_weights ./weights/svhn/resnet18/svhn_resnet18_fsrTest.pth \
--fsr_weights ./weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth
python test.py --load_name svhn_resnet18 --dataset svhn --model resnet18 --device 7 \
--backbone_weights ./weights/svhn/resnet18/svhn_resnet18_fsrTest.pth \
--fsr_weights ./weights/mnist/resnet18/mnist_resnet18_fsrTest.pth
python test.py --load_name svhn_resnet18 --dataset svhn --model resnet18 --device 7 \
--backbone_weights ./weights/svhn/resnet18/svhn_resnet18_fsrTest.pth \
--fsr_weights ./weights/svhn/resnet18/svhn_resnet18_fsrTest.pth


# # Source data: tinyimg
# python test.py --load_name tinyimg_resnet18 --dataset tinyimg --model resnet18 --device 7 \
# --backbone_weights ./weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth \
# --fsr_weights ./weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth
# python test.py --load_name tinyimg_resnet18 --dataset tinyimg --model resnet18 --device 7 \
# --backbone_weights ./weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth \
# --fsr_weights ./weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth
# python test.py --load_name tinyimg_resnet18 --dataset tinyimg --model resnet18 --device 7 \
# --backbone_weights ./weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth \
# --fsr_weights ./weights/mnist/resnet18/mnist_resnet18_fsrTest.pth
# python test.py --load_name tinyimg_resnet18 --dataset tinyimg --model resnet18 --device 7 \
# --backbone_weights ./weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth \
# --fsr_weights ./weights/svhn/resnet18/svhn_resnet18_fsrTest.pth

# # Target data: tinyimg
# python test.py --load_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 7 \
# --backbone_weights ./weights/cifar10/resnet18/cifar10_resnet18_fsrTest.pth \
# --fsr_weights ./weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth
# python test.py --load_name cifar100_resnet18 --dataset cifar100 --model resnet18 --device 7 \
# --backbone_weights ./weights/cifar100/resnet18/cifar100_resnet18_fsrTest.pth \
# --fsr_weights ./weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth
# python test.py --load_name mnist_resnet18 --dataset mnist --model resnet18 --device 7 \
# --backbone_weights ./weights/mnist/resnet18/mnist_resnet18_fsrTest.pth \
# --fsr_weights ./weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth
# python test.py --load_name svhn_resnet18 --dataset svhn --model resnet18 --device 7 \
# --backbone_weights ./weights/svhn/resnet18/svhn_resnet18_fsrTest.pth \
# --fsr_weights ./weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth
# python test.py --load_name tinyimg_resnet18 --dataset tinyimg --model resnet18 --device 7 \
# --backbone_weights ./weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth \
# --fsr_weights ./weights/tinyimg/resnet18/tinyimg_resnet18_fsrTest.pth