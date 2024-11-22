# Source data: cifar10
python test.py --load_name cifar10_vgg16 --dataset cifar10 --model vgg16 --device 7 \
--backbone_weights ./weights/cifar10/vgg16/cifar10_vgg16_fsrTest.pth \
--fsr_weights ./weights/cifar10/vgg16/cifar10_vgg16_fsrTest.pth

python test.py --load_name cifar10_vgg16 --dataset cifar10 --model vgg16 --device 7 \
--backbone_weights ./weights/cifar10/vgg16/cifar10_vgg16_fsrTest.pth \
--fsr_weights ./weights/cifar100/vgg16/cifar100_vgg16_fsrTest.pth

python test.py --load_name cifar10_vgg16 --dataset cifar10 --model vgg16 --device 7 \
--backbone_weights ./weights/cifar10/vgg16/cifar10_vgg16_fsrTest.pth \
--fsr_weights ./weights/mnist/vgg16/mnist_vgg16_fsrTest.pth

python test.py --load_name cifar10_vgg --dataset cifar10 --model vgg16 --device 7 \
--backbone_weights ./weights/cifar10/vgg16/cifar10_vgg16_fsrTest.pth \
--fsr_weights ./weights/svhn/vgg16/svhn_vgg16_fsrTest.pth


# Source data: cifar100
python test.py --load_name cifar100_vgg16 --dataset cifar100 --model vgg16 --device 7 \
--backbone_weights ./weights/cifar100/vgg16/cifar100_vgg16_fsrTest.pth \
--fsr_weights ./weights/cifar10/vgg16/cifar10_vgg16_fsrTest.pth
python test.py --load_name cifar100_vgg16 --dataset cifar100 --model vgg16 --device 7 \
--backbone_weights ./weights/cifar100/vgg16/cifar100_vgg16_fsrTest.pth \
--fsr_weights ./weights/cifar100/vgg16/cifar100_vgg16_fsrTest.pth
python test.py --load_name cifar100_vgg16 --dataset cifar100 --model vgg16 --device 7 \
--backbone_weights ./weights/cifar100/vgg16/cifar100_vgg16_fsrTest.pth \
--fsr_weights ./weights/mnist/vgg16/mnist_vgg16_fsrTest.pth
python test.py --load_name cifar100_vgg16 --dataset cifar100 --model vgg16 --device 7 \
--backbone_weights ./weights/cifar100/vgg16/cifar100_vgg16_fsrTest.pth \
--fsr_weights ./weights/svhn/vgg16/svhn_vgg16_fsrTest.pth

# Source data: mnist
python test.py --load_name mnist_vgg16 --dataset mnist --model vgg16 --device 7 \
--backbone_weights ./weights/mnist/vgg16/mnist_vgg16_fsrTest.pth \
--fsr_weights ./weights/cifar10/vgg16/cifar10_vgg16_fsrTest.pth
python test.py --load_name mnist_vgg16 --dataset mnist --model vgg16 --device 7 \
--backbone_weights ./weights/mnist/vgg16/mnist_vgg16_fsrTest.pth \
--fsr_weights ./weights/cifar100/vgg16/cifar100_vgg16_fsrTest.pth
python test.py --load_name mnist_vgg16 --dataset mnist --model vgg16 --device 7 \
--backbone_weights ./weights/mnist/vgg16/mnist_vgg16_fsrTest.pth \
--fsr_weights ./weights/mnist/vgg16/mnist_vgg16_fsrTest.pth
python test.py --load_name mnist_vgg16 --dataset mnist --model vgg16 --device 7 \
--backbone_weights ./weights/mnist/vgg16/mnist_vgg16_fsrTest.pth \
--fsr_weights ./weights/svhn/vgg16/svhn_vgg16_fsrTest.pth

# Source data: svhn
python test.py --load_name svhn_vgg16 --dataset svhn --model vgg16 --device 7 \
--backbone_weights ./weights/svhn/vgg16/svhn_vgg16_fsrTest.pth \
--fsr_weights ./weights/cifar10/vgg16/cifar10_vgg16_fsrTest.pth
python test.py --load_name svhn_vgg16 --dataset svhn --model vgg16 --device 7 \
--backbone_weights ./weights/svhn/vgg16/svhn_vgg16_fsrTest.pth \
--fsr_weights ./weights/cifar100/vgg16/cifar100_vgg16_fsrTest.pth
python test.py --load_name svhn_vgg16 --dataset svhn --model vgg16 --device 7 \
--backbone_weights ./weights/svhn/vgg16/svhn_vgg16_fsrTest.pth \
--fsr_weights ./weights/mnist/vgg16/mnist_vgg16_fsrTest.pth
python test.py --load_name svhn_vgg16 --dataset svhn --model vgg16 --device 7 \
--backbone_weights ./weights/svhn/vgg16/svhn_vgg16_fsrTest.pth \
--fsr_weights ./weights/svhn/vgg16/svhn_vgg16_fsrTest.pth
