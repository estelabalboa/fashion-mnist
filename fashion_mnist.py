# -*- coding: utf-8 -*-


print (" Installing FastAI libraries... (takes 2 min) ")
# !pip install fastai==0.7.0 > /dev/null
print ("\n Installing required libraries...")
# !pip install torchtext==0.2.3 > /dev/null
# !git clone https://github.com/fastai/fastai.git fastai_ml
# !ln -s fastai_ml/courses/ml1/fastai/ fastai

from fastai.imports import *
from fastai.io import *
from fastai.torch_imports import *
import torch

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# !git clone https://github.com/zalandoresearch/fashion-mnist.git

from fastai.imports import *
from fastai.io import *
from fastai.torch_imports import *
import torch

# !gunzip /content/fashion-mnist/data/fashion/t*-ubyte.gz

from mlxtend.data import loadlocal_mnist

x, y = loadlocal_mnist(
  images_path = 'fashion-mnist/data/fashion/train-images-idx3-ubyte',
  labels_path = 'fashion-mnist/data/fashion/train-labels-idx1-ubyte')
# y = np.array(y, dtype = 'int')

y = np.array(y, dtype = 'int')

type(y[0])

mean = x.mean()

display(mean)

std = x.std()
x=(x-mean)/std
mean, std, x.mean(), x.std()

# Validacion

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print(X_train[0])

from fastai.metrics import *
from fastai.model import *
from fastai.dataset import *

import torch.nn as nn

net = nn.Sequential(
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.LogSoftmax()
)

path = 'fashion-mnist/data/fashion'
md = ImageClassifierData.from_arrays(path, (X_train,y_train), (X_test, y_test))

loss = nn.NLLLoss()
metrics = [accuracy]
# solemos ver el accuracy en el test
opt = optim.Adam(net.parameters())

fit(net, md, n_epochs=20, crit=loss, opt=opt, metrics = metrics)
# Cada vez que hacemos el fit, el modelo se entrena y se almacena en su propia variable de modelo net

