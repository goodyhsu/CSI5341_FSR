# Source data: cifar10
python test.py --load_name cifar10_wideresnet34 --dataset cifar10 --model wideresnet34 --device 7 \
--backbone_weights ./weights/cifar10/wideresnet34/cifar10_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/cifar10/wideresnet34/cifar10_wideresnet34_fsrTest.pth

python test.py --load_name cifar10_wideresnet34 --dataset cifar10 --model wideresnet34 --device 7 \
--backbone_weights ./weights/cifar10/wideresnet34/cifar10_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/cifar100/wideresnet34/cifar100_wideresnet34_fsrTest.pth

python test.py --load_name cifar10_wideresnet34 --dataset cifar10 --model wideresnet34 --device 7 \
--backbone_weights ./weights/cifar10/wideresnet34/cifar10_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/mnist/wideresnet34/mnist_wideresnet34_fsrTest.pth

python test.py --load_name cifar10_vgg --dataset cifar10 --model wideresnet34 --device 7 \
--backbone_weights ./weights/cifar10/wideresnet34/cifar10_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/svhn/wideresnet34/svhn_wideresnet34_fsrTest.pth


# Source data: cifar100
python test.py --load_name cifar100_wideresnet34 --dataset cifar100 --model wideresnet34 --device 7 \
--backbone_weights ./weights/cifar100/wideresnet34/cifar100_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/cifar10/wideresnet34/cifar10_wideresnet34_fsrTest.pth
python test.py --load_name cifar100_wideresnet34 --dataset cifar100 --model wideresnet34 --device 7 \
--backbone_weights ./weights/cifar100/wideresnet34/cifar100_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/cifar100/wideresnet34/cifar100_wideresnet34_fsrTest.pth
python test.py --load_name cifar100_wideresnet34 --dataset cifar100 --model wideresnet34 --device 7 \
--backbone_weights ./weights/cifar100/wideresnet34/cifar100_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/mnist/wideresnet34/mnist_wideresnet34_fsrTest.pth
python test.py --load_name cifar100_wideresnet34 --dataset cifar100 --model wideresnet34 --device 7 \
--backbone_weights ./weights/cifar100/wideresnet34/cifar100_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/svhn/wideresnet34/svhn_wideresnet34_fsrTest.pth

# Source data: mnist
python test.py --load_name mnist_wideresnet34 --dataset mnist --model wideresnet34 --device 7 \
--backbone_weights ./weights/mnist/wideresnet34/mnist_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/cifar10/wideresnet34/cifar10_wideresnet34_fsrTest.pth
python test.py --load_name mnist_wideresnet34 --dataset mnist --model wideresnet34 --device 7 \
--backbone_weights ./weights/mnist/wideresnet34/mnist_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/cifar100/wideresnet34/cifar100_wideresnet34_fsrTest.pth
python test.py --load_name mnist_wideresnet34 --dataset mnist --model wideresnet34 --device 7 \
--backbone_weights ./weights/mnist/wideresnet34/mnist_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/mnist/wideresnet34/mnist_wideresnet34_fsrTest.pth
python test.py --load_name mnist_wideresnet34 --dataset mnist --model wideresnet34 --device 7 \
--backbone_weights ./weights/mnist/wideresnet34/mnist_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/svhn/wideresnet34/svhn_wideresnet34_fsrTest.pth

# Source data: svhn
python test.py --load_name svhn_wideresnet34 --dataset svhn --model wideresnet34 --device 7 \
--backbone_weights ./weights/svhn/wideresnet34/svhn_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/cifar10/wideresnet34/cifar10_wideresnet34_fsrTest.pth
python test.py --load_name svhn_wideresnet34 --dataset svhn --model wideresnet34 --device 7 \
--backbone_weights ./weights/svhn/wideresnet34/svhn_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/cifar100/wideresnet34/cifar100_wideresnet34_fsrTest.pth
python test.py --load_name svhn_wideresnet34 --dataset svhn --model wideresnet34 --device 7 \
--backbone_weights ./weights/svhn/wideresnet34/svhn_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/mnist/wideresnet34/mnist_wideresnet34_fsrTest.pth
python test.py --load_name svhn_wideresnet34 --dataset svhn --model wideresnet34 --device 7 \
--backbone_weights ./weights/svhn/wideresnet34/svhn_wideresnet34_fsrTest.pth \
--fsr_weights ./weights/svhn/wideresnet34/svhn_wideresnet34_fsrTest.pth
