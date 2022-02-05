import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv




class Decoder(nn.Module):

	def __init__(self, batch_size):
		super().__init__()
		self.Nh = 128
		self.batch_size = batch_size
		self.conv1 = nn.ConvTranspose3d(in_channels=self.Nh, out_channels=128, kernel_size=3, padding=1)
		self.conv2 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
		self.conv3 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
		self.conv4 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
		self.conv5 = nn.ConvTranspose3d(in_channels=32, out_channels=2, kernel_size=3, padding=1)
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=2, mode='nearest')
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = F.interpolate(x, scale_factor=2, mode='nearest')
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = F.interpolate(x, scale_factor=2, mode='nearest')
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		x = self.softmax(x)
		return x