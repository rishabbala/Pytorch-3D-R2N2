import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv




class Encoder(nn.Module):


	def __init__(self):
		super().__init__()
		self.resize = tv.transforms.Resize(127)
		self.conv1 = nn.Conv2d(in_channels=4, out_channels=96, kernel_size=7, padding=3)
		self.conv2 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(num_features=128, affine=False)
		self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
		self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(num_features=256, affine=False)
		self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.bn3 = nn.BatchNorm2d(num_features=256, affine=False)
		self.linear = nn.Linear(in_features=256, out_features=1024)
		self.pool = nn.MaxPool2d(kernel_size=2)
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
		self.tanh = nn.Tanh()


	def forward(self, x):
		x = self.resize(x)
		x = self.conv1(x)
		x = self.pool(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.bn1(x)
		x = self.pool(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.pool(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.bn2(x)
		x = self.pool(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.pool(x)
		x = self.leaky_relu(x)
		x = self.conv6(x)
		x = self.bn3(x)
		x = self.pool(x)
		x = self.leaky_relu(x)
		# x = torch.flatten(x, start_dim=1)
		x = x.view(x.shape[0], x.shape[1])
		x = self.linear(x)
		x = self.leaky_relu(x)
		# print(x)
		# exit()
		return x