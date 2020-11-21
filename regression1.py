import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from hyperdash import Experiment

import matplotlib.pyplot as plt
import rotem_helpers
import numpy as np
import imageio

torch.manual_seed(1)  # reproducible
exp = Experiment("tennis_regression")

x_train = rotem_helpers.load_obj('/Users/rotemisraeli/Documents/python/tennis/dataset2/x_train.np')
y_train = rotem_helpers.load_obj('/Users/rotemisraeli/Documents/python/tennis/dataset2/y_train.np')
x_train = torch.from_numpy(np.asarray(x_train,dtype=float)).float()
y_train = torch.from_numpy(np.asarray(y_train,dtype=float)).float()
x_test = rotem_helpers.load_obj('/Users/rotemisraeli/Documents/python/tennis/dataset2/x_test.np')
y_test = rotem_helpers.load_obj('/Users/rotemisraeli/Documents/python/tennis/dataset2/y_test.np')
x_test = torch.from_numpy(np.asarray(x_test,dtype=float)).float()
y_test = torch.from_numpy(np.asarray(y_test,dtype=float)).float()


# another way to define a network
net = torch.nn.Sequential(
	torch.nn.Linear(8, 100),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(100, 200),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(200, 300),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(300, 400),
	torch.nn.LeakyReLU(),
	torch.nn.Linear(400, 90),
)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 64
EPOCH = 200

torch_dataset_train = Data.TensorDataset(x_train, y_train)
loader_train = Data.DataLoader(
	dataset=torch_dataset_train,
	batch_size=BATCH_SIZE,
	shuffle=True, num_workers=2, )

torch_dataset_test = Data.TensorDataset(x_test, y_test)
loader_test = Data.DataLoader(
	dataset=torch_dataset_test,
	batch_size=BATCH_SIZE,
	shuffle=True, num_workers=2, )


def calc_accuricy():
	error = 0
	for step, (batch_x, batch_y) in enumerate(loader_test):  # for each training step
		b_x = Variable(batch_x)
		b_y = Variable(batch_y)
		prediction = net(b_x)
		loss = loss_func(prediction, b_y)  # must be (1. nn output, 2. target)
		error+=float(loss)

	exp.metric('accuracy' , error)

# start training
for epoch in range(EPOCH):
	exp.metric('epoch',epoch)
	for step, (batch_x, batch_y) in enumerate(loader_train):  # for each training step

		b_x = Variable(batch_x)
		b_y = Variable(batch_y)

		prediction = net(b_x)  # input x and predict based on x
		loss = loss_func(prediction, b_y)  # must be (1. nn output, 2. target)

		if  float(loss) < 15000:
			exp.metric('loss',float(loss))
			calc_accuricy()
		optimizer.zero_grad()  # clear gradients for next train
		loss.backward()  # backpropagation, compute gradients
		optimizer.step()  # apply gradients

torch.save(net, 'regression2.torch')

exp.end()


