export CUDA_VISIBLE_DEVICES=5

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting02" ]; then
    mkdir ./logs/LongForecasting02
fi
seq_len=512
model_name=TimeCapsule

root_path_name=../dataset/weather/
data_path_name=(weather.csv)
model_id_name=(weather)
data_name=(custom)

random_seed=2021

# weather
# for pred_len in 96
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name[0]} \
#       --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name[0]} \
#       --features M \
#       --revin 1\
#       --jepa 1\
#       --n_block 1\
#       --level_dim 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --d_compress 4 8 4\
#       --n_heads 4 \
#       --d_model 256 \
#       --d_ff 512 \
#       --dropout 0.3\
#       --gamma 0.1\
#       --des 'Exp' \
#       --train_epochs 15 \
#       --itr 1 --batch_size 128 --learning_rate 5e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done

# for pred_len in 192 
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name[0]} \
#       --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name[0]} \
#       --features M \
#       --revin 1\
#       --jepa 1\
#       --n_block 1\
#       --level_dim 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --d_compress 4 8 4\
#       --n_heads 4 \
#       --d_model 256 \
#       --d_ff 512 \
#       --dropout 0.5\
#       --gamma 0.3\
#       --des 'Exp' \
#       --train_epochs 15 \
#       --itr 1 --batch_size 128 --learning_rate 3e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done

# for pred_len in 336
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name[0]} \
#       --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name[0]} \
#       --features M \
#       --revin 1\
#       --jepa 1\
#       --n_block 1\
#       --level_dim 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --d_compress 4 8 4\
#       --n_heads 4 \
#       --d_model 256 \
#       --d_ff 512 \
#       --dropout 0.5\
#       --gamma 0.3\
#       --des 'Exp' \
#       --train_epochs 15 \
#       --itr 1 --batch_size 128 --learning_rate 3e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done

# for pred_len in 720 
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name[0]} \
#       --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name[0]} \
#       --features M \
#       --revin 1\
#       --jepa 1 \
#       --n_block 2\
#       --level_dim 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --d_compress 4 8 4\
#       --n_heads 4 \
#       --d_model 256 \
#       --d_ff 512 \
#       --dropout 0.7 \
#       --gamma 0.7 \
#       --des 'Exp' \
#       --train_epochs 10 \
#       --itr 1 --batch_size 128 --learning_rate 3e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done