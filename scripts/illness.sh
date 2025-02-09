export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting02" ]; then
    mkdir ./logs/LongForecasting02
fi
seq_len=96
model_name=TimeCapsule

root_path_name=../dataset/illness/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2021

# ILI
for pred_len in 24
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name[0]} \
      --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name[0]} \
      --features M \
      --jepa 1\
      --revin 1\
      --n_block 1\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --n_heads 2 \
      --d_compress 8 8 2\
      --d_model 512 \
      --d_ff  256\
      --dropout 0.2\
      --patience 20\
      --des 'Exp' \
      --train_epochs 50 \
      --gamma 0.7\
      --itr 1 --batch_size 64 --learning_rate 2e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 36 48 60
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name[0]} \
      --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name[0]} \
      --features M \
      --revin 1\
      --jepa 1\
      --n_block 2\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --n_heads 2 \
      --d_compress 8 8 2\
      --d_model 512 \
      --d_ff  256\
      --dropout 0.2\
      --patience 20\
      --des 'Exp' \
      --train_epochs 40 \
      --gamma 0.8\
      --itr 1 --batch_size 32 --learning_rate 4e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
