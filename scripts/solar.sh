export CUDA_VISIBLE_DEVICES=2

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting02" ]; then
    mkdir ./logs/LongForecasting02
fi
seq_len=512
model_name=TimeCapsule

root_path_name=../dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=solar
data_name=Solar

random_seed=2021

#Solar
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
      --jepa 0\
      --revin 1\
      --n_block 1\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --n_heads 4 \
      --d_compress 4 8 4\
      --d_model 256 \
      --d_ff 512 \
      --gamma 0.8 \
      --dropout 0.2\
      --des 'Exp' \
      --patience 10\
      --train_epochs 50 \
      --itr 1 --batch_size 64 --learning_rate 2e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --jepa 1 \
      --revin 1\
      --n_block 1\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --n_heads 4 \
      --d_compress 4 8 4\
      --d_model 256 \
      --d_ff 512 \
      --gamma 0.8 \
      --dropout 0.2\
      --des 'Exp' \
      --patience 10\
      --train_epochs 50 \
      --itr 1 --batch_size 64 --learning_rate 2e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --jepa 1 \
      --revin 1\
      --n_block 2\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --n_heads 4 \
      --d_compress 4 8 4\
      --d_model 256 \
      --d_ff 512 \
      --gamma 0.8 \
      --dropout 0.2\
      --des 'Exp' \
      --patience 10\
      --train_epochs 50 \
      --itr 1 --batch_size 64 --learning_rate 2e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

# (4, 8, 4) will cause JEPA to fail convergence
for pred_len in  720  
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
      --jepa 1 \
      --revin 1\
      --n_block 2\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --n_heads 4 \
      --d_compress 1 8 1\
      --d_model 256 \
      --d_ff 512 \
      --gamma 0.8 \
      --dropout 0.2\
      --des 'Exp' \
      --patience 10\
      --train_epochs 50 \
      --itr 1 --batch_size 64 --learning_rate 2e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
