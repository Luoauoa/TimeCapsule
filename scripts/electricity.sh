export CUDA_VISIBLE_DEVICES=3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting02" ]; then
    mkdir ./logs/LongForecasting02
fi
seq_len=512
model_name=TimeCapsule

root_path_name=../dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2021

# electricity
for pred_len in 96 
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
      --n_heads 4 \
      --d_compress 4 8 4\
      --d_model 128 \
      --d_ff 2048 \
      --dropout 0.1\
      --des 'Exp' \
      --train_epochs 50 \
      --patience 15 \
      --gamma 0.7\
      --itr 1 --batch_size 32 --learning_rate 1e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 192
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
      --n_heads 4 \
      --d_compress 4 8 4\
      --d_model 64 \
      --d_ff 2048 \
      --dropout 0.1\
      --des 'Exp' \
      --train_epochs 50 \
      --gamma 0.7\
      --itr 1 --batch_size 32 --learning_rate 8e-4 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 336
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
      --n_heads 4 \
      --d_compress 4 8 4\
      --d_model 64 \
      --d_ff 512 \
      --dropout 0.1\
      --des 'Exp' \
      --train_epochs 50 \
      --gamma 0.7\
      --itr 1 --batch_size 32 --learning_rate 8e-4 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 720
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
      --jepa 0\
      --revin 1\
      --n_block 0\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --n_heads 4 \
      --d_compress 1 8 1\
      --d_model 128 \
      --d_ff 1024 \
      --dropout 0.2\
      --des 'Exp' \
      --train_epochs 50 \
      --gamma 0.2\
      --itr 1 --batch_size 32 --learning_rate 5e-4 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
