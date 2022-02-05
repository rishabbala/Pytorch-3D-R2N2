import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from three_d_lstm import Three_Dim_LSTM
import torchvision as tv
import matplotlib.pyplot as plt
from decoder import Decoder
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from os import walk
import random
import glob
from torch.utils.data import DataLoader
import binvox_rw
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
from torch.optim.lr_scheduler import ExponentialLR




def get_class_of_images(batch_size):
	fname = ''
	dirs = os.listdir('ShapeNetRendering')
	dirs.remove('rendering_only.tgz')
	# fname = random.choice(dirs)
	fname = dirs[0]
	folder_name = 'ShapeNetRendering/' + fname + '/'
	sfname_list = []
	for i in range(batch_size):
		sfname = ''
		while sfname in sfname_list or sfname == '':
			sfname = random.choice(os.listdir(folder_name)[:20])
			# sfname = os.listdir(folder_name)[0]
		sfname_list.append(sfname)
	subfolder_name = [folder_name + sfname for sfname in sfname_list]
	subfolder_name = [name + '/rendering' for name in subfolder_name]
	max_num_image = 10000
	for i in range(batch_size):
		image_names = glob.glob(subfolder_name[i]+'/*.png')
		num_images = len(image_names)
		if num_images < max_num_image:
			max_num_image = num_images
	num_ip_images = random.randint(1, max_num_image)
	print(subfolder_name)
	exit()
	return subfolder_name, num_ip_images, fname, sfname_list



def create_dataset(num_ip_images, subfolder_name):
	for j in range(len(subfolder_name)):
		image_input_names = glob.glob(subfolder_name[j]+'/*.png')
		name = random.sample(image_input_names, num_ip_images)
		for i in range(num_ip_images):
			img = tv.io.read_image(name[i])
			if i == 0:
				image = img.unsqueeze(0)
			else:
				image = torch.cat((image, img.unsqueeze(0)), axis=0)
		if j == 0:
			input_images = image.unsqueeze(0)
		else:
			input_images = torch.cat((input_images, image.unsqueeze(0)), axis=0)
	input_images.requires_grad=False
	return input_images.float().to(device)



def get_voxel_outputs(fname, sfname_list):
	pth = 'ShapeNetVox32/'+fname+'/'
	pth = [pth + sfname for sfname in sfname_list]
	try:
		for i in range(len(sfname_list)):
			with open(pth[i]+'/model.binvox', 'rb') as f:
				model = binvox_rw.read_as_3d_array(f)
			if i == 0:
				output = np.expand_dims(model.data, axis=0)
			else:
				output = np.concatenate((output, np.expand_dims(model.data, axis=0)), axis=0)
		return output, 1
	except:
		return 1, -1


def train(writer):
	batch_size = 5
	min_loss = 100000
	Nh = 128

	NLLLoss = nn.NLLLoss()
	torch.autograd.set_detect_anomaly(True)

	encoder_net = Encoder().to(device)
	three_d_lstm_net = Three_Dim_LSTM(batch_size).to(device)
	decoder_net = Decoder(batch_size).to(device)

	hidden_states = torch.tensor(np.zeros((batch_size, Nh, 4, 4, 4)), dtype=torch.float32, device=device)

	encoder_net.load_state_dict(torch.load('encoder_net.pth'))
	three_d_lstm_net.load_state_dict(torch.load('three_d_lstm_net.pth'))
	decoder_net.load_state_dict(torch.load('decoder_net.pth'))

	optim = Adam([{'params': encoder_net.parameters()},{'params': three_d_lstm_net.parameters()},{'params': decoder_net.parameters()}])
	# scheduler = ExponentialLR(optim, gamma=0.99)
	# encoder_optim = Adam(encoder_net.parameters())
	# lstm_optim = Adam(three_d_lstm_net.parameters())
	# decoder_optim = Adam(decoder_net.parameters())

	for n_iter in range(60000):
		hidden_states = hidden_states.detach()
		subfolder_name, num_ip_images, fname, sfname_list = get_class_of_images(batch_size)
		dataset = create_dataset(num_ip_images, subfolder_name)
		vox_model, flag = get_voxel_outputs(fname, sfname_list)
		if flag == -1:
			continue
		output = np.where(vox_model == False, 0, 1)
		output = torch.tensor(output, requires_grad=False, device=device).long().to(device)

		optim.zero_grad()
		# encoder_optim.zero_grad()
		# lstm_optim.zero_grad()
		# decoder_optim.zero_grad()
		for i in range(num_ip_images):
			encoder_op = encoder_net(dataset[:, i, :, :, :])
			hidden_states = three_d_lstm_net(encoder_op, hidden_states)
		pred = decoder_net(hidden_states)
		loss = NLLLoss(pred, output)
		loss.backward()
		total_loss = loss.item()

		optim.step()
		# scheduler.step()
		# decoder_optim.step()
		# lstm_optim.step()
		# encoder_optim.step()

		if (total_loss < min_loss):
			min_loss = total_loss
			torch.save(encoder_net.state_dict(), 'encoder_net.pth')
			torch.save(three_d_lstm_net.state_dict(), 'three_d_lstm_net.pth')
			torch.save(decoder_net.state_dict(), 'decoder_net.pth')
		
		if n_iter%1000 == 0:
			torch.save(encoder_net.state_dict(), 'encoder_net_latest.pth')
			torch.save(three_d_lstm_net.state_dict(), 'three_d_lstm_net_latest.pth')
			torch.save(decoder_net.state_dict(), 'decoder_net_latest.pth')

		writer.add_scalar('Loss', total_loss, n_iter)
		print(f"Iteration: {n_iter}, Loss: {total_loss}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
writer = SummaryWriter('runs/exp_4p2')
train(writer)