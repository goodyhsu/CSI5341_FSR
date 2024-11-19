# Impact of Data Domain on FSR Training

## Performance of Different Backbones / FSR
Row: backbone, column: FSR.
The values in the cells are in the represented as average accuracies using 4 different adversarial attack schemes.

|          | CIFAR10  | CIFAR100 | MNIST    | SVHN     |TinyImageNet |
|----------|----------|----------|----------|----------|----------|
| CIFAR10  | 0.8317 / 0.5175  | 0.8315 / 0.4978   |0.8056 / 0.4796    | 0.8289 / 0.4986  | NaN / NaN |
| CIFAR100 | 0.5803 / 0.2440   | 0.5830 / 0.2580   | 0.5678 / 0.2422   | 0.5708 / 0.2361  | NaN / NaN |
| MNIST    | 0.9941 / 0.9899  | 0.9932 / 0.9893  | 0.9960 / 0.9926  | 0.9934 / 0.9895  | NaN / NaN |
| SVHN     | 0.9236 / 0.5384  | 0.9225 / 0.5447  | 0.9062 / 0.5024  | 0.9238 / 0.5865  | NaN / NaN |
| TinyImageNet  | NaN / NaN | NaN / NaN | NaN / NaN | NaN / NaN | NaN / NaN |

### Observations
* 

## TODOs
- [ ] Inference & fill result table
- [ ] Transfer learning: add another layer / replace the final layer in the FSR module