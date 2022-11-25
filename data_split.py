'''
./data/ -> ./data/train/
        -> ./data/valid/
        -> ./data/test/
give data to ./data/ then run this code
then data will be splitted
'''

from os import listdir
import random
import os
import shutil

file_list = listdir('./data')

random.shuffle(file_list)

train_rate = 0.8

data_num = int(len(file_list)*train_rate)

test_num = len(file_list) - data_num
train_num = int(data_num*train_rate)
valid_num = data_num - train_num

train_data = file_list[:train_num]
valid_data = file_list[train_num:train_num+valid_num]
test_data = file_list[train_num+valid_num:]

train_path = './data/train/'
valid_path = './data/valid/'
test_path = './data/test/'

if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(valid_path):
    os.makedirs(valid_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)

for f in train_data:
    shutil.move('./data/'+f, train_path+f)
for f in valid_data:
    shutil.move('./data/'+f, valid_path+f)
for f in test_data:
    shutil.move('./data/'+f, test_path+f)
