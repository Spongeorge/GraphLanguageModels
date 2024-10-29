#!/bin/bash

graph_representation=gGLM  # set list gGLM lGLM
dataset_construction=random
reset_params=False # False True (set this to True for gGLM and lGLM to reset the parameters, i.e., to run the Graph Transformer baselines gGT and lGT)
num_epochs=50 # 50
modelsize=t5-small # t5-small t5-base t5-large
train_batch_size=32
gradient_accumulation_steps=1
eos_usage=False # bidirectional
init_additional_buckets_from=1e6
save_model_filepath=/home/students/kolber/Investigating-GLM-hidden-states/checkpoints
#load_model_filepath=/checkpoints/best_epoch
load_hf_model_filepath=plenz/GLM-t5-small
early_stopping=20

device=cuda
logging_level=INFO


for params_to_train in all # all head
    do
        for radius in 1 # 1 2 3 4 5
        do
            if [ $radius -eq 4 ]
            then
                num_maskeds="0 1 2 3 4 5"
            else
                num_maskeds="0"
            fi
            for num_masked in $num_maskeds
            do
                if [ $params_to_train == "all" ]
                then
                    learning_rate=0.0001
                else
                    learning_rate=0.005
                fi
                echo ""
                echo running $params_to_train $radius $num_masked $seed $graph_representation $load_model_filepath
                python experiments/encoder/relation_prediction/train_LM.py \
                    --num_masked $num_masked \
                    --radius $radius \
                    --params_to_train $params_to_train \
                    --learning_rate $learning_rate \
                    --graph_representation $graph_representation \
                    --dataset_construction $dataset_construction \
                    --reset_params $reset_params \
                    --num_epochs $num_epochs \
                    --modelsize $modelsize \
                    --train_batch_size $train_batch_size \
                    --gradient_accumulation_steps $gradient_accumulation_steps \
                    --eval_batch_size $train_batch_size \
                    --eos_usage $eos_usage \
                    --init_additional_buckets_from $init_additional_buckets_from \
                    --device $device \
                    --logging_level $logging_level \
                    --load_hf_model_filepath $load_hf_model_filepath \
                    --num_additional_buckets 3 \
                    --early_stopping $early_stopping \
                    --save_model_filepath $save_model_filepath #\
#                    --load_model_filepath $load_model_filepath 
                echo done $params_to_train $radius $num_masked $seed $graph_representation
                echo ""
            done
        done
    done
done
