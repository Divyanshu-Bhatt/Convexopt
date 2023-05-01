#!/bin/sh
func=combination_function

# Data Generation
data_dim=1
domain=(-10 10)
parent_folder="$func"_"$data_dim"
num_points=10000

python datagenerator.py --function $func --data_dim $data_dim --num_points $num_points --force 1 --low ${domain[0]} --high ${domain[1]} --common_loc $parent_folder

epsilon=1e-3
data_dir=$func
device=mps

# Artifical Neural Network
model_save_dir=MLP
lr=2e-3
epochs=5
evaluation_period=1

python main.py --epsilon $epsilon --model_save_dir $model_save_dir --data_dir $data_dir --lr $lr --device $device --epochs $epochs --evaluation_period $evaluation_period --common_loc $parent_folder
python optimum_value.py --network_dir $model_save_dir --common_loc $parent_folder

# Radial Basis Function
epochs=2
evaluation_period=1
batch_size=64
lr=2e-3
python rbf.py --epsilon $epsilon --common_loc $parent_folder --data_dir $data_dir --epochs $epochs --evaluation_period $evaluation_period --device $device --lr $lr 

# Polynomial Regression
degree=2
epochs=2
evaluation_period=1
lr=2e-3
python polynomial.py --epsilon $epsilon --common_loc $parent_folder --data_dir $data_dir --epochs $epochs --evaluation_period $evaluation_period --device $device --lr $lr --degree $degree
