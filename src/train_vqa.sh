NGPU=4 python -m torch.distributed.launch --nproc_per_node=4 train_KAT.py \
  --train_data VQA/KAT/okvqa/train2014 \
  --eval_data VQA/KAT/okvqa/val2014 \
  --model_size large \
  --lr 0.00003 \
  --optim adamw \
  --scheduler linear \
  --weight_decay 0.01 \
  --text_maxlength 64 \
  --per_gpu_batch_size 1 \
  --n_context 40 \
  --total_step 8000 \
  --warmup_step 1000 \
  --name gptj1 \
  --checkpoint_dir VQA/KAT/checkpoint \
  --accumulation_steps 1 \
  --use_gpt --gpt_name codexr


python train_unifiedQA.py \
  --train_data VQA/KAT/unifiedQA/train2014 \
  --eval_data VQA/KAT/unifiedQA/val2014 \
  --model_size base \
  --lr 0.00003 \
  --optim adamw \
  --scheduler linear \
  --weight_decay 0.01 \
  --text_maxlength 64 \
  --per_gpu_batch_size 4 \
  --n_context 40 \
  --total_step 20000 \
  --warmup_step 1000 \
  --name base_noCuqa \
  --checkpoint_dir VQA/KAT/checkpoint \
  --accumulation_steps 1 \
  --gpt_name codexr
