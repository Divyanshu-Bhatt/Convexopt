#!/bin/sh
func=combination_function

# Data Generation
# data_dim=1
dimension=(2 10 100)
domain=(-2 2)
num_points=10000

for data_dim in ${dimension[@]};
    do
    parent_folder="$func"_"$data_dim"
    python datagenerator.py --function $func --data_dim $data_dim --num_points $num_points --force 1 --low ${domain[0]} --high ${domain[1]} --common_loc $parent_folder

    epsilon=1e-3
    data_dir=$func
    device=cpu

    # Artifical Neural Network
    model_save_dir=MLP"$data_dim"
    lr=2e-3
    epochs=100
    evaluation_period=10

    python main.py --epsilon $epsilon --model_save_dir $model_save_dir --data_dir $data_dir --lr $lr --device $device --epochs $epochs --evaluation_period $evaluation_period --common_loc $parent_folder
    python optimum_value.py --network_dir $model_save_dir --common_loc $parent_folder

    # Radial Basis Function
    epochs=100
    evaluation_period=10
    batch_size=64
    lr=0.01
    python rbf.py --epsilon $epsilon --common_loc $parent_folder --data_dir $data_dir --epochs $epochs --evaluation_period $evaluation_period --device $device --lr $lr 

    # Polynomial Regression
    # degree=10
    # epochs=100
    # evaluation_period=10
    # lr=2e-3
    # python polynomial.py --epsilon $epsilon --common_loc $parent_folder --data_dir $data_dir --epochs $epochs --evaluation_period $evaluation_period --device $device --lr $lr --degree $degree
    done