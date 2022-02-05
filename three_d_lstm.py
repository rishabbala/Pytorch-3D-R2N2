import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from einops import repeat



class Three_Dim_LSTM(nn.Module):


	def __init__(self, batch_size):
		super().__init__()
		self.batch_size = batch_size
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.Nh = 128

		self.conv3d_z = nn.Conv3d(in_channels=self.Nh, out_channels=self.Nh, kernel_size=3, padding=(1, 1, 1), padding_mode='zeros', bias=False)
		self.conv3d_r = nn.Conv3d(in_channels=self.Nh, out_channels=self.Nh, kernel_size=3, padding=(1, 1, 1), padding_mode='zeros', bias=False)
		self.conv3d_m = nn.Conv3d(in_channels=self.Nh, out_channels=self.Nh, kernel_size=3, padding=(1, 1, 1), padding_mode='zeros', bias=False)
		
		self.bias_z = nn.Parameter(torch.tensor(np.random.rand(1, self.Nh, 4, 4, 4), dtype=torch.float), requires_grad=True)
		self.bias_r = nn.Parameter(torch.tensor(np.random.rand(1, self.Nh, 4, 4, 4), dtype=torch.float), requires_grad=True)
		self.bias_m = nn.Parameter(torch.tensor(np.random.rand(1, self.Nh, 4, 4, 4), dtype=torch.float), requires_grad=True)

		self.lin_z = nn.Linear(in_features=1024, out_features=self.Nh*4*4*4, device=self.device, bias=False)
		self.lin_r = nn.Linear(in_features=1024, out_features=self.Nh*4*4*4, device=self.device, bias=False)
		self.lin_m = nn.Linear(in_features=1024, out_features=self.Nh*4*4*4, device=self.device, bias=False)
		
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()



	def forward(self, x, hidden_states):
		# x = repeat(inp, 'n c -> n 4 4 4 c')
		lz = self.lin_z(x)
		lz = lz.view(self.batch_size, self.Nh, 4, 4, 4)
		# lz = torch.permute(lz, (0, 4, 1, 2, 3))
		z = self.sigmoid(self.conv3d_z(hidden_states) + lz + self.bias_z)

		lr = self.lin_r(x)
		lr = lr.view(self.batch_size, self.Nh, 4, 4, 4)
		# lr = torch.permute(lr, (0, 4, 1, 2, 3))
		r = self.sigmoid(self.conv3d_r(hidden_states) + lr + self.bias_r)

		lm = self.lin_m(x)
		lm = lm.view(self.batch_size, self.Nh, 4, 4, 4)
		# lm = torch.permute(lm, (0, 4, 1, 2, 3))
		mem = self.tanh(self.conv3d_m(r*hidden_states) + lm + self.bias_m)
		
		hidden_states = (torch.ones(z.shape).to(self.device)-z)*hidden_states + z*mem
		return hidden_states