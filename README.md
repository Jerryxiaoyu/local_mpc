# MAML implementation in Tensorflow


*This code is still being developed and subject to change.

## Prerequisites
- Python 2.7
- [Tensorflow 1.4.0](https://github.com/tensorflow/tensorflow/tree/r1.4)
- matplotlib 2.1.0
- [NumPy](http://www.numpy.org/) 1.13.3
- tqdm 4.19.4
- [colorlog](https://github.com/borntyping/python-colorlog) 3.1.0

## Usage

### Regression

Train 5-shot regreesion model:
```bash
python main.py --dataset sin --K 5 --num_updates 1 --norm None --is_train
```

Details about the training FLAGs
```
--K: draw K samples as meta-train and meta-val
--model_type: for regression, I only use fully connected layers
--loss_type: for regression, I use MeanSquareError as loss criterion
--num_updates: do `num_updates` graident step for meta-step
--norm: use batch_norm or not
--alpha: learning rate for meta-train (same notation as the paper)
--beta: learning rate for meta-val (same notation as the paper)
--is_train: speficy a training phase
```

Evalaute the model (either specify the directory of the checkpoint or the checkpoint itself):
```bash
python main.py --dataset sin --K 5 --num_updates 5 --norm None --restore_checkpoint PATH_TO_CHECKPOINT
```

- [3] Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

