#!/bin/sh
func=l1_norm
model_save_dir_prefix=MLP

lr=1e-4
epochs=250
evaluation_period=5
device=mps

common_loc_prefix=combination_function

hidden_dims1=(64 128)
hidden_dims2=(32 64)
dimension=(10 15 20)

for data_dim in ${dimension[@]};
    do
        parent_folder="$common_loc_prefix"_"$data_dim"
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