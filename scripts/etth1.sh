export CUDA_VISIBLE_DEVICES=2

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=TimeCapsule

root_path_name=../dataset/ETT-small/
data_path_name=(ETTh1.csv ETTm2.csv)
model_id_name=(ETTh1 ETTm2)
data_name=(ETTh1 ETTm2)

random_seed=2021

#ETTh1
# set d_ff to 512 or even 256 can achieve similar results with much lower cost
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
#       --jepa 1 \
#       --revin 1\
#       --n_block 0\
#       --level_dim 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --n_heads 1 \
#       --d_compress 1 8 1\
#       --d_model 16 \
#       --d_ff  512 \
#       --gamma 0.3 \
#       --dropout 0.6\
#       --des 'Exp' \
#       --patience 10\
#       --train_epochs 20 \
#       --itr 1 --batch_size 64 --learning_rate 6e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
#       --jepa 1 \
#       --revin 1\
#       --n_block 1\
#       --level_dim 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --n_heads 1 \
#       --d_compress 1 8 1\
#       --d_model 16 \
#       --d_ff  512 \
#       --gamma 0.3 \
#       --dropout 0.6\
#       --des 'Exp' \
#       --patience 10\
#       --train_epochs 20 \
#       --itr 1 --batch_size 64 --learning_rate 5e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done

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
      --n_block 1\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --n_heads 1 \
      --d_compress 1 8 1\
      --d_model 64 \
      --d_ff  1024 \
      --gamma 0.3 \
      --dropout 0.6\
      --des 'Exp' \
      --patience 10\
      --train_epochs 20 \
      --itr 1 --batch_size 64 --learning_rate 2e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --jepa 0 \
      --revin 1\
      --n_block 1\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --n_heads 16 \
      --d_compress 4 8 4\
      --d_model 64 \
      --d_ff  1024 \
      --gamma 0.3 \
      --dropout 0.9\
      --des 'Exp' \
      --patience 10\
      --train_epochs 20 \
      --itr 1 --batch_size 64 --learning_rate 2e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
