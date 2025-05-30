export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=TimeCapsule

root_path_name=../dataset/ETT-small/
data_path_name=(ETTh2.csv ETTm1.csv)
model_id_name=(ETTh2 ETTm1)
data_name=(ETTh2 ETTm1)

random_seed=2021

# ETTh2
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
      --n_heads 2 \
      --d_compress 4 16 4\
      --d_model 128 \
      --d_ff 256 \
      --gamma 0.3 \
      --dropout 0.7\
      --des 'Exp' \
      --patience 10\
      --train_epochs 20 \
      --itr 1 --batch_size 32 --learning_rate 2e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --n_heads 2 \
      --d_compress 4 16 4\
      --d_model 64 \
      --d_ff 256 \
      --gamma 0.3 \
      --dropout 0.7\
      --des 'Exp' \
      --patience 10\
      --train_epochs 20 \
      --itr 1 --batch_size 32 --learning_rate 1e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --n_heads 2 \
      --d_compress 8 32 4\
      --d_model 128 \
      --d_ff 256 \
      --gamma 0.3 \
      --dropout 0.7\
      --des 'Exp' \
      --patience 10\
      --train_epochs 20 \
      --itr 1 --batch_size 32 --learning_rate 1e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --jepa 1\
      --revin 1\
      --n_block 1\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --n_heads 2 \
      --d_compress 8 32 4\
      --d_model 256 \
      --d_ff  512 \
      --gamma 0.1 \
      --dropout 0.6\
      --des 'Exp' \
      --patience 10\
      --train_epochs 5 \
      --itr 1 --batch_size 32 --learning_rate 1e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
