#!/bin/sh
func=combination_function
model_save_dir_prefix=MLP2

lr=2e-3
epochs=100
evaluation_period=5
device=mps

hidden_dims1=(64 128)
hidden_dims2=(32 64)
dimension=(1 5 10)

for data_dim in ${dimension[@]};
    do
        parent_folder="$func"_"$data_dim"
        python datagenerator.py --function $func --force 1 --common_loc $parent_folder --data_dim $data_dim
        for dims1 in ${hidden_dims1[@]};
            do
            for dims2 in ${hidden_dims2[@]};
                do 
                    common_loc="$parent_folder"/dims_"$dims1"_"$dims2"
                    model_save_dir="$model_save_dir_prefix"_"$data_dim"_"$dims1"_"$dims2"
                    python main.py --hidden_dims $dims1 $dims2 --model_save_dir $model_save_dir --data_dir $func --lr $lr --device $device --epochs $epochs --evaluation_period $evaluation_period --common_loc $common_loc
                    python optimum_value.py --network_dir $model_save_dir --common_loc $common_loc
                done
            done
    done