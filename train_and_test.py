import numpy as np
import torch
import torch.utils.data as Data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch.nn.functional as F
import time
import os
import pickle
from RAAL_model import RAAL_model

def main():
	result_unit="s"
	x_1=np.load("./data_files/"+result_unit+"_data_np/data_du_list.npy")
	x_2=np.load("./data_files/"+result_unit+"_data_np/data_du_2_list.npy")
	x_3=np.load("./data_files/"+result_unit+"_data_np/data_tree_word_np.npy")
	x_5=np.load("./data_files/"+result_unit+"_data_np/data_1.npy")
	y=np.load("./data_files/"+result_unit+"_data_np/data_result.npy")
	embedding_weight=torch.FloatTensor(np.load("./data_files/"+result_unit+"_data_np/embedding_weight.npy"))

	train_num=int(100000*0.8)
	print("Training data:", train_num,"item，Test data:", 100000-train_num, "item")
	train_x_1=torch.Tensor(x_1[:train_num])
	train_x_2=torch.Tensor(x_2[:train_num])
	train_x_3=torch.LongTensor(x_3[:train_num])
	train_x_5=torch.Tensor(x_5[:train_num])
	train_y=torch.Tensor(y[:train_num]).unsqueeze(-1)

	test_x_1=torch.Tensor(x_1[train_num:])
	test_x_2=torch.Tensor(x_2[train_num:])
	test_x_3=torch.LongTensor(x_3[train_num:])
	test_x_5=torch.Tensor(x_5[train_num:])
	test_y=torch.Tensor(y[train_num:]).unsqueeze(-1)

	dict_param={
		"embedding_dim":32,
		"lstm_num_layer":1,
		"lstm_hidden_dim":16,
		"nn_1_hidden_dim":(train_x_3.shape)[1]*((train_x_1.shape)[2]+1+16),
		"nn_2_hidden_dim":32+(train_x_5.shape)[1]+(train_x_5.shape)[1],
		"nn_1_out_num":32,
		"nn_2_out_num":8,
		"NN_1_num_layer":3,
		"NN_2_num_layer":2,
		"lr":0.001,
		"test_batch_size":200,
		"train_batch_size":30,
		"epochs":500,
		"test_num_epoch":5,
		"is_bidirectional":False,
		"optim":"Adam",
		"loss":"MSELoss",
		"save_model_path":"./model.pkl",
	}
	print(dict_param["nn_1_hidden_dim"])
	print(train_x_1.shape)
	print(train_x_2.shape)
	print(train_x_3.shape)
	print(train_x_5.shape)
	if dict_param["loss"]=="MSELoss":
		loss=torch.nn.MSELoss(reduce=True, size_average=True)
	elif dict_param["loss"]=="SmoothL1Loss":
		loss=torch.nn.SmoothL1Loss()
	elif dict_param["loss"]=="L1Loss":
		loss=torch.nn.L1Loss()

	# Initialising the model
	if torch.cuda.is_available():
		device = "cuda:0"
		print("This model training has been accelerated using the GPU.")
		model=RAAL_model(dict_param, embedding_weight).to(device)
		embedding_weight=embedding_weight.to(device)
		train_x_1=train_x_1.to(device)
		train_x_2=train_x_2.to(device)
		train_x_3=train_x_3.to(device)
		train_x_5=train_x_5.to(device)
		train_y=train_y.to(device)

		test_x_1=test_x_1.to(device)
		test_x_2=test_x_2.to(device)
		test_x_3=test_x_3.to(device)
		test_x_5=test_x_5.to(device)
		test_y=test_y.to(device)

		loss=loss.to(device)
	else:
		model=RAAL_model(dict_param, embedding_weight)

	# Random batches of data
	train_dataset=Data.TensorDataset(train_x_1, train_x_2, train_x_3,  train_x_5, train_y)
	train_loader=Data.DataLoader(
		dataset=train_dataset,
		batch_size=dict_param["train_batch_size"],
		shuffle=True,
		num_workers=0,
		)

	test_dataset=Data.TensorDataset(test_x_1, test_x_2, test_x_3,  test_x_5, test_y)
	test_loader=Data.DataLoader(
		dataset=test_dataset,
		batch_size=dict_param["test_batch_size"],
		shuffle=False,
		num_workers=0,
		)

	# Optimizers
	if dict_param["optim"]=="SGD":
		optimizer=torch.optim.SGD(model.parameters(), lr=dict_param["lr"], momentum=0.9)
	elif dict_param["optim"]=="SGD_Momentum":
		optimizer=torch.optim.SGD(model.parameters(),lr=dict_param["lr"],momentum=0.8,nesterov=True)
	elif dict_param["optim"]=="Adagrad":
		optimizer=torch.optim.Adagrad(model.parameters(), lr=dict_param["lr"], lr_decay=0, weight_decay=0, initial_accumulator_value=0)
	elif dict_param["optim"]=="RMSprop":
		optimizer=torch.optim.RMSprop(model.parameters(), lr=dict_param["lr"], alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
	elif dict_param["optim"]=="Adam":
		optimizer=torch.optim.Adam(model.parameters(), lr=dict_param["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

	loss_epoch_list=[]
	loss_value_list=[]
	test_epoch_list=[]
	test_mse_list=[]
	test_mase_list=[]
	test_r2_list=[]
	test_relative_error_list=[]
	test_cor_relation_list=[]

	log_str="Output data units："+str(result_unit)+"\nOptimizers："+str(dict_param["optim"])+"\nLoss function："+str(dict_param["loss"])+"\nBatch data size："+str(dict_param["train_batch_size"])+"\nTraining iterations："+str(dict_param["epochs"])
	print(log_str)

	# Model training 
	print("RAAL_Model starts training......")
	model.train()
	for epoch in range(dict_param["epochs"]):
		loss_list=[]
		for step, (batch_x_1, batch_x_2, batch_x_3, batch_x_5, batch_y) in enumerate(train_loader):
			optimizer.zero_grad()
			out=model(batch_x_1, batch_x_2, batch_x_3, batch_x_5, dict_param)			
			los=loss(out, batch_y)
			loss_list.append(los.item())
			los.backward()
			optimizer.step()
		l=np.array(loss_list).mean()
		loss_value_list.append(l)
		loss_epoch_list.append(epoch)
		train_log="epoch:"+str(epoch)+"\tloss:"+str(l)
		log_str+="\n"+train_log
		print(train_log)

		# Start testing
		if (epoch+1)%dict_param["test_num_epoch"]==0:
			mse, mase, r2, relative_error, cor_relation=model_test(model, test_loader, dict_param)
			test_epoch_list.append(epoch)
			test_mse_list.append(mse)
			test_mase_list.append(mase)
			test_r2_list.append(r2)
			test_relative_error_list.append(relative_error)
			test_cor_relation_list.append(cor_relation)
			test_log="\n\tModel testing：\tMSE:"+str(mse)+"\tMASE:"+str(mase)+"\tr2:"+str(r2)+"\n\t\t\tRelative Error:"+str(relative_error)+"\tcor_relation:"+str(cor_relation)
			print(test_log)
			log_str+=test_log
	plot_data_write_into_file(loss_epoch_list, loss_value_list, test_epoch_list, test_mse_list, test_mase_list, test_r2_list, test_relative_error_list, test_cor_relation_list, log_str)

	torch.save(model, dict_param["save_model_path"])

def write_pkl(path, data):
	output=open(path, "wb")
	pickle.dump(data, output)
	output.close()

def plot_data_write_into_file(loss_epoch_list, loss_value_list, test_epoch_list, test_mse_list, test_mase_list, test_r2_list, test_relative_error_list, test_cor_relation_list, log_str):
	if os.path.exists("./result") == False:
		os.mkdir("./result/")
	t=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
	if os.path.exists("./result/"+t) == False:
		os.mkdir("./result/"+t)
	write_pkl("./result/"+t+"/loss_epoch_list.pkl", loss_epoch_list)
	write_pkl("./result/"+t+"/loss_value_list.pkl", loss_value_list)
	write_pkl("./result/"+t+"/test_epoch_list.pkl", test_epoch_list)
	write_pkl("./result/"+t+"/test_mse_list.pkl", test_mse_list)
	write_pkl("./result/"+t+"/test_mase_list.pkl", test_mase_list)
	write_pkl("./result/"+t+"/test_r2_list.pkl", test_r2_list)
	write_pkl("./result/"+t+"/test_relative_error_list.pkl", test_relative_error_list)
	write_pkl("./result/"+t+"/test_cor_relation_list.pkl", test_cor_relation_list)
	with open("./result/"+t+"/log.txt", "w") as f:
		f.write(log_str)

def relative_error_fun(x, y):
	return np.mean(np.abs(x-y)/y)

def cor_relation_fun(x, y):
	x_mean=np.mean(x)
	y_mean=np.mean(y)
	return np.sum((x-x_mean)*(y-y_mean))/np.sqrt(np.sum((x-x_mean)*(x-x_mean))*np.sum((y-y_mean)*(y-y_mean)))

def model_test(model, test_loader, dict_param):
	model.eval()
	mse_list=[]
	mase_list=[]
	r2_list=[]
	relative_error_list=[]
	cor_relation_list=[]
	mse=0
	mase=0
	r2=0
	relative_error=0
	cor_relation=0

	for step, (batch_x_1, batch_x_2, batch_x_3, batch_x_5, batch_y) in enumerate(test_loader):
		out=model(batch_x_1, batch_x_2, batch_x_3, batch_x_5, dict_param)
		mse_list.append(mean_squared_error(batch_y.cpu().detach().numpy(), out.cpu().detach().numpy()))
		mase_list.append(mean_absolute_error(batch_y.cpu().detach().numpy(), out.cpu().detach().numpy()))
		r2_list.append(r2_score(batch_y.cpu().detach().numpy(), out.cpu().detach().numpy()))
		relative_error_list.append(relative_error_fun(batch_y.cpu().detach().numpy(), out.cpu().detach().numpy()))
		cor_relation_list.append(cor_relation_fun(batch_y.cpu().detach().numpy(), out.cpu().detach().numpy()))
	mse=np.array(mse_list).mean()
	mase=np.array(mase_list).mean()
	r2=np.array(r2_list).mean()
	relative_error=np.array(relative_error_list).mean()
	cor_relation=np.array(cor_relation_list).mean()
	model.train()
	return mse, mase, r2, relative_error, cor_relation

if __name__ == '__main__':
	main()