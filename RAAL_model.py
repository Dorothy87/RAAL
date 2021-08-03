import torch
import torch.nn.functional as F
import math
import time
import os
import pickle

class RAAL_model(torch.nn.Module):
	def __init__(self, dict_param, embedding_weight):
		super(RAAL_model, self).__init__()
		self.word_embeddings=torch.nn.Embedding.from_pretrained(embedding_weight, freeze=False)
		self.LSTM=torch.nn.LSTM(
			input_size=dict_param["embedding_dim"], 
			hidden_size=dict_param["lstm_hidden_dim"], 
			num_layers=dict_param["lstm_num_layer"], 
			batch_first=True, 
			bidirectional=dict_param["is_bidirectional"],
		)
		self.w_1 = torch.nn.Parameter(torch.Tensor(dict_param["lstm_hidden_dim"], dict_param["lstm_hidden_dim"]))
		self.u_1 = torch.nn.Parameter(torch.Tensor(dict_param["lstm_hidden_dim"], 1))
		self.w_2 = torch.nn.Parameter(torch.Tensor(28, 28))
		self.u_2 = torch.nn.Parameter(torch.Tensor(28, 1))
		self.line=torch.nn.Sequential(
			torch.nn.Linear(18*16, 28),
			torch.nn.BatchNorm1d(28),
			torch.nn.LeakyReLU(inplace=True))

		if dict_param["NN_1_num_layer"]==1:
			self.NN_1=torch.nn.Sequential(
				torch.nn.Linear(dict_param["nn_1_hidden_dim"], dict_param["nn_1_out_num"]),
				torch.nn.BatchNorm1d(dict_param["nn_1_out_num"]),
				torch.nn.LeakyReLU(inplace=True)
			)
		elif dict_param["NN_1_num_layer"]==2:
			self.NN_1=torch.nn.Sequential(
				torch.nn.Linear(dict_param["nn_1_hidden_dim"], dict_param["nn_1_out_num"]),
				torch.nn.BatchNorm1d(dict_param["nn_1_out_num"]),
				torch.nn.LeakyReLU(inplace=True),
				torch.nn.Linear(dict_param["nn_1_out_num"], dict_param["nn_1_out_num"]),
				torch.nn.BatchNorm1d(dict_param["nn_1_out_num"]),
				torch.nn.LeakyReLU(inplace=True)
			)
		elif dict_param["NN_1_num_layer"]==3:
			self.NN_1=torch.nn.Sequential(
				torch.nn.Linear(dict_param["nn_1_hidden_dim"], dict_param["nn_1_out_num"]),
				torch.nn.BatchNorm1d(dict_param["nn_1_out_num"]),
				torch.nn.LeakyReLU(inplace=True),
				torch.nn.Linear(dict_param["nn_1_out_num"], dict_param["nn_1_out_num"]),
				torch.nn.BatchNorm1d(dict_param["nn_1_out_num"]),
				torch.nn.LeakyReLU(inplace=True),
				torch.nn.Linear(dict_param["nn_1_out_num"], dict_param["nn_1_out_num"]),
				torch.nn.BatchNorm1d(dict_param["nn_1_out_num"]),
				torch.nn.LeakyReLU(inplace=True)
			)
		if dict_param["NN_2_num_layer"]==1:
			self.NN_2=torch.nn.Sequential(
				torch.nn.Linear(dict_param["nn_2_hidden_dim"], dict_param["nn_2_out_num"]),
				torch.nn.BatchNorm1d(dict_param["nn_2_out_num"]),
				torch.nn.LeakyReLU(inplace=True)
			)
		elif dict_param["NN_2_num_layer"]==2:
			self.NN_2=torch.nn.Sequential(
				torch.nn.Linear(dict_param["nn_2_hidden_dim"], dict_param["nn_2_out_num"]),
				torch.nn.LeakyReLU(inplace=True),
				torch.nn.Linear(dict_param["nn_2_out_num"], dict_param["nn_2_out_num"]),
				torch.nn.LeakyReLU(inplace=True)
			)
		elif dict_param["NN_2_num_layer"]==3:
			self.NN_2=torch.nn.Sequential(
				torch.nn.Linear(dict_param["nn_2_hidden_dim"], dict_param["nn_2_out_num"]),
				torch.nn.BatchNorm1d(dict_param["nn_2_out_num"]),
				torch.nn.LeakyReLU(inplace=True),
				torch.nn.Linear(dict_param["nn_2_out_num"], dict_param["nn_2_out_num"]),
				torch.nn.BatchNorm1d(dict_param["nn_2_out_num"]),
				torch.nn.LeakyReLU(inplace=True),
				torch.nn.Linear(dict_param["nn_2_out_num"], dict_param["nn_2_out_num"]),
				torch.nn.LeakyReLU(inplace=True)
			)
		self.NN_3=torch.nn.Sequential(
			torch.nn.Linear(dict_param["nn_2_out_num"], 16),
			torch.nn.LeakyReLU(inplace=True))
		self.out_layer=torch.nn.Linear(16, 1)

	def forward(self, x_1, x_2, x_3, x_5, dict_param):
		batch_size=(x_3.shape)[0]
		sentence_size=(x_3.shape)[1]
		x_3=self.word_embeddings(x_3.view(batch_size*sentence_size, (x_3.shape)[2]))
		x_3, (h_n, c_n)=self.LSTM(x_3)
		h_n=torch.transpose(h_n, 0, 1).contiguous().view(batch_size, sentence_size, -1)
		h_n_t=h_n.view(batch_size, 18*16)
		u_1_te = torch.tanh(torch.matmul(h_n, self.w_1))
		a_1 = torch.matmul(u_1_te, self.u_1)
		a_1_s = F.softmax(a_1, dim=1)
		h_n = h_n * a_1_s

		u_2_te=torch.tanh(torch.matmul(x_5, self.w_2))
		a_2=torch.matmul(u_2_te, self.u_2)
		a_2_s=F.softmax(a_2, dim=1)
		x_5=x_5*a_2_s
		h_n_t=self.line(h_n_t)
		h_n_t=h_n_t.mul(x_5)
		x=torch.cat((x_1, x_2.unsqueeze(-1), h_n), -1).view(batch_size, dict_param["nn_1_hidden_dim"])
		x=self.NN_1(x)
		x=torch.cat((x, x_5, h_n_t), -1)
		x=self.NN_2(x)
		x=self.NN_3(x)
		out=self.out_layer(x)
		return out