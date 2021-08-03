import csv
import math
import numpy as np
import string
import locale
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
import gensim
import pickle
import os
from pathlib import Path


NUMBER_CONSTANT = {0:"zero ", 1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven",
                8:"eight", 9:"nine", 10:"ten", 11:"eleven", 12:"twelve", 13:"thirteen",
                14:"fourteen", 15:"fifteen", 16:"sixteen", 17:"seventeen", 18:"eighteen", 19:"nineteen" }
IN_HUNDRED_CONSTANT = {2:"twenty", 3:"thirty", 4:"forty", 5:"fifty", 6:"sixty", 7:"seventy", 8:"eighty", 9:"ninety"}
BASE_CONSTANT = {0:" ", 1:"hundred", 2:"thousand", 3:"million", 4:"billion"}
#supported number range is 1-n billion
def translateNumberToEnglish(number):
    if str(number).isnumeric():
        if str(number)[0] == '0' and len(str(number)) > 1:
            return translateNumberToEnglish(int(number[1:]))
        if int(number) < 20:
            return NUMBER_CONSTANT[int(number)]
        elif int(number) < 100:
            if str(number)[1] == '0':
                return IN_HUNDRED_CONSTANT[int(str(number)[0])]
            else:
                return IN_HUNDRED_CONSTANT[int(str(number)[0])] + " - " + NUMBER_CONSTANT[int(str(number)[1])]
        else:
            locale.setlocale(locale.LC_ALL, "English_United States.1252")
            strNumber = locale.format("%d"    , number, grouping=True)
            numberArray = str(strNumber).split(",")
            stringResult = ""
            groupCount = len(numberArray) + 1
            for groupNumber in numberArray:
                if groupCount > 1 and groupNumber[0:] != "000":
                    stringResult += str(getUnderThreeNumberString(str(groupNumber))) + " "
                else:
                    break
                groupCount -= 1
                if groupCount > 1:
                    stringResult += BASE_CONSTANT[groupCount] + " "
            endPoint = len(stringResult) - len(" hundred,")
            return stringResult
                
    else:
        print("please input a number!")

#between 0-999
def getUnderThreeNumberString(number):
    if str(number).isnumeric() and len(number) < 4:
        if len(number) < 3:
            return translateNumberToEnglish(int(number))
        elif len(number) == 3 and number[0:] == "000":
            return " "
        elif len(number) == 3 and number[1:] == "00":
            return NUMBER_CONSTANT[int(number[0])] + "  " + BASE_CONSTANT[1]
        else:    
            return NUMBER_CONSTANT[int(number[0])] + "  " + BASE_CONSTANT[1] + " and " + translateNumberToEnglish((number[1:]))
    else:
        print("number must below 1000")

if Path("./data_files").is_dir() == False:
	os.mkdir("./data_files")

path="./data_files"


word2vec_size=32
result_unit="s"
print("The resultant data units used in this instance are：", result_unit)

data_temp_1_1=[]
data_temp_2=[]
data_temp_3=[]
with open('sqldataset.csv', 'r') as f:
	reader = csv.reader(f)
	for d in reader:
		a=[int(d[i+3]) for i in range(28)]
		data_temp_1_1.append(a)
		data_temp_2.append(d[0])
		if result_unit=="ms":
			data_temp_3.append(float(int(d[2])))
		else:
			data_temp_3.append(float(int(d[2])/1000.0))

if Path(path+"/"+result_unit+"_data_np").is_dir() == False:
	os.mkdir(path+"/"+result_unit+"_data_np")
np.save(path+"/"+result_unit+"_data_np/data_result.npy", np.array(data_temp_3))
print("np.array(data_temp_3).shape:", np.array(data_temp_3).shape)

data_temp_1 =[]
for i in range(len(data_temp_1_1[0])):
	b =[]
	for j in range(len(data_temp_1_1)):
		b.append(data_temp_1_1[j][i])
	data_temp_1.append(b)

data_1=[]
average_list=[]
total_list=[0]*28
for i in range(len(data_temp_1)):
	average_list.append(float(sum(data_temp_1[i]))/len(data_temp_1[i]))
	for value in data_temp_1[i]:
		total_list[i]+=float(value-average_list[i])**2
	stddev = math.sqrt(total_list[i]/len(data_temp_1[i]))
	if stddev!=0:
		data_1.append([(x-average_list[i])/stddev for x in data_temp_1[i]])
	else:
		data_1.append([0 for x in data_temp_1[i]])
np.save(path+"/"+result_unit+"_data_np/data_1.npy", np.array(data_1).T)
print("np.array(data_1).shape:", np.array(data_1).T.shape)

data_du_list=[]
data_du_2_list=[]
data_word_list=[]
for i in range(len(data_temp_2)):
	temp_1=data_temp_2[i].strip().split("\n")
	du_1_temp=[]
	du_2_temp=[]
	word_temp=[]
	for j in range(len(temp_1)):
		n=temp_1[j].find("]")
		du_1_temp.append(eval(temp_1[j][0:n+1]))
		if temp_1[j][n+2]!="-":
			du_2_temp.append(float(temp_1[j][n+2]))
			word_temp.append(temp_1[j][n+4:])
		else:
			du_2_temp.append(-1.0)
			word_temp.append(temp_1[j][n+5:])
	
	data_du_list.append(du_1_temp)
	data_du_2_list.append(du_2_temp)
	data_word_list.append(word_temp)

max_1=16
max_2=20
for i in range(len(data_du_list)):
	l=len(data_du_list[i])
	for j in range(len(data_du_list[i])):
		l=len(data_du_list[i][j])

data_du_np=np.zeros((len(data_du_list), max_1, max_2))
data_du_2_np=np.zeros((len(data_du_list), max_1))
for i in range(len(data_du_list)):
	for j in range(len(data_du_list[i])):
		if j==max_1:
			break
		for n in range(len(data_du_list[i][j])):
			if n==max_2:
				break
			data_du_np[i][j][n]=data_du_list[i][j][n]
		data_du_2_np[i][j]=data_du_2_list[i][j]

print("data_du_np.shape:",data_du_np.shape)
print("data_du_2_np.shape:",data_du_2_np.shape)
np.save(path+"/"+result_unit+"_data_np/data_du_list.npy", data_du_np)
np.save(path+"/"+result_unit+"_data_np/data_du_2_list.npy", data_du_2_np)

data_tree_word_list=[]
data_tree_word_line=""
word_punctuation=string.punctuation.replace("_", "")
for i in range(len(data_word_list)):
	data_list_temp=[]
	t_line=""
	for j in range(len(data_word_list[i])):
		
		temp_list=data_word_list[i][j].split("\n")
		d=""
		for m in temp_list:
			for nn in m:
				if nn in word_punctuation:
					d=d+" "+nn+" "
				else:
					d=d+nn
		t_list=d.replace(" &  & ", "&&").strip().split()
		t_str=""
		for n in range(len(t_list)):
			if t_list[n][0] in string.digits:
				t_str+=" "+translateNumberToEnglish(int(t_list[n]))
			else:
				t_str+=" "+t_list[n]
		data_line=' '.join(t_str.strip().split())
		data_list_temp.append(t_str.strip().split())
		t_line+=data_line+"\n"
	data_tree_word_line+=t_line+"\n"
	data_tree_word_list.append(data_list_temp)

with open(path+"/word_line.txt", "w") as f:
	f.write(data_tree_word_line)

if Path(path+"/"+result_unit+"_data_pkl").is_dir() == False:
	os.mkdir(path+"/"+result_unit+"_data_pkl")

output = open(path+'/'+result_unit+'_data_pkl/data_tree_word_list.pkl', 'wb')
pickle.dump(data_tree_word_list, output)
output.close()

pkl_file = open(path+'/'+result_unit+'_data_pkl/data_tree_word_list.pkl', 'rb')
data_tree_word_list = pickle.load(pkl_file)
pkl_file.close()

print("\nStart training the word vector model......")
import logging 
import os.path 
import sys 
import multiprocessing 

from gensim.models import Word2Vec 
from gensim.models.word2vec import LineSentence 

size=32
window=10
min_count=1
iters=2

inp="./data_files/word_line.txt"
outp="corpus.model"
outp2="corpus.vector"
model = Word2Vec(LineSentence(inp),size=size,window=window,min_count=min_count,workers=6,iter=iters) 
model.save("./data_files/"+str(size)+"_"+outp)
model.wv.save_word2vec_format("./data_files/"+str(size)+"_"+outp2)
model = gensim.models.Word2Vec.load(path+'/'+str(word2vec_size)+'_corpus.model')
with open(path+"/word_line.txt", "r") as f:
	data_list=f.readlines()

word_dic={}
num=1
word_num_dic={}
for i in range(len(data_list)):
	temp_line=data_list[i].strip().split()
	for j in range(len(temp_line)):
		if temp_line[j].strip() in word_num_dic.keys():

			word_dic[temp_line[j].strip()]+=1
		else:
			word_dic[temp_line[j].strip()]=1
			word_num_dic[temp_line[j].strip()]=num
			num+=1

num_word_dic = dict(zip(word_num_dic.values(),word_num_dic.keys()))

output = open(path+'/'+result_unit+'_data_pkl/word_num_dic.pkl', 'wb')
pickle.dump(word_num_dic, output)
output.close()

output = open(path+'/'+result_unit+'_data_pkl/num_word_dic.pkl', 'wb')
pickle.dump(num_word_dic, output)
output.close()

output = open(path+'/'+result_unit+'_data_pkl/word_dic.pkl', 'wb')
pickle.dump(word_dic, output)
output.close()

embedding_weight_list=[]
embedding_weight_list.append(np.zeros((32)))
for key in num_word_dic.keys():
	embedding_weight_list.append(model[num_word_dic[key]])

np.save(path+"/"+result_unit+"_data_np/embedding_weight.npy", np.array(embedding_weight_list))

max_level_1=0
max_level_2=max_1
max_level_3=0
for i in range(len(data_tree_word_list)):
	if i+1>max_level_1:
		max_level_1=i+1
	for j in range(len(data_tree_word_list[i])):
		for n in range(len(data_tree_word_list[i][j])):
			if n+1> max_level_3:
				max_level_3=n+1

data_tree_word_np=np.zeros((max_level_1, max_level_2, max_level_3))
print("data_tree_word_np.shape:",data_tree_word_np.shape)
np.save(path+"/"+result_unit+"_data_np/data_tree_word_np.npy", data_tree_word_np)