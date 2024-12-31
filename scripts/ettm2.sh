export CUDA_VISIBLE_DEVICES=4

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting03" ]; then
    mkdir ./logs/LongForecasting03
fi
seq_len=512
model_name=TimeCapsule

root_path_name=../dataset/ETT-small/
data_path_name=(ETTh1.csv ETTm2.csv)
model_id_name=(ETTh1 ETTm2)
data_name=(ETTh1 ETTm2)

random_seed=2021

# ETTm2
# for pred_len in 96 
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name[1]} \
#       --model_id ${model_id_name[1]}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name[1]} \
#       --features M \
#       --jepa 1\
#       --revin 1\
#       --n_block 1\
#       --level_dim 1\
#       --d_compress 1 8 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --n_heads 2 \
#       --d_model 256 \
#       --d_ff 512 \
#       --dropout 0.7\
#       --gamma 0.3\
#       --des 'Exp' \
#       --train_epochs 30 \
#       --itr 1 --batch_size 128 --learning_rate 1.5e-3 >logs/LongForecasting03/$model_name'_'${model_id_name[1]}'_'$seq_len'_'$pred_len.log 
# done

# for pred_len in 192 
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name[1]} \
#       --model_id ${model_id_name[1]}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name[1]} \
#       --features M \
#       --jepa 1\
#       --revin 1\
#       --n_block 1\
#       --level_dim 1\
#       --d_compress 4 8 4\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --n_heads 2 \
#       --d_model 256 \
#       --d_ff 512 \
#       --dropout 0.7\
#       --gamma 0.3\
#       --des 'Exp' \
#       --train_epochs 30 \
#       --itr 1 --batch_size 128 --learning_rate 1.5e-3 >logs/LongForecasting03/$model_name'_'${model_id_name[1]}'_'$seq_len'_'$pred_len.log 
# done

# for pred_len in 336 
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name[1]} \
#       --model_id ${model_id_name[1]}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name[1]} \
#       --features M \
#       --jepa 1\
#       --revin 1\
#       --n_block 2\
#       --level_dim 1\
#       --d_compress 4 8 4\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --n_heads 2 \
#       --d_model 256 \
#       --d_ff 512 \
#       --dropout 0.7\
#       --gamma 0.8\
#       --patience 10\
#       --des 'Exp' \
#       --train_epochs 30 \
#       --itr 1 --batch_size 128 --learning_rate 8e-4 >logs/LongForecasting03/$model_name'_'${model_id_name[1]}'_'$seq_len'_'$pred_len.log 
# done

# for pred_len in 720 
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name[1]} \
#       --model_id ${model_id_name[1]}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name[1]} \
#       --features M \
#       --jepa 1\
#       --revin 1\
#       --n_block 2\
#       --level_dim 1\
#       --d_compress 4 8 4\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --n_heads 2 \
#       --d_model 256 \
#       --d_ff 512 \
#       --dropout 0.7\
#       --gamma 0.7\
#       --patience 10\
#       --des 'Exp' \
#       --train_epochs 30 \
#       --itr 1 --batch_size 128 --learning_rate 1e-3 >logs/LongForecasting03/$model_name'_'${model_id_name[1]}'_'$seq_len'_'$pred_len.log 
# done