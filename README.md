# ESACL: Enhanced Seq2Seq Autoencoder via Contrastive Learning for AbstractiveText Summarization
This repo is for our paper "Enhanced Seq2Seq Autoencoder via Contrastive Learning for AbstractiveText Summarization". Our program is building on top of the Huggingface ```transformers``` framework. You can refer to their repo at: https://github.com/huggingface/transformers/tree/master/examples/seq2seq.

## Local Setup
Tested with Python 3.7 via virtual environment. Clone the repo, go to the repo folder, setup the virtual environment, and install the required packages:
```bash
$ python3.7 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Install ```apex```
Based on the recommendation from HuggingFace, both finetuning and eval are 30% faster with ```--fp16```. For that you need to install ```apex```.
```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Data
Create a directory for data used in this work named ```data```:
```bash
$ mkdir data
```

### CNN/DM
```bash
$ wget https://cdn-datasets.huggingface.co/summarization/cnn_dm_v2.tgz
$ tar -xzvf cnn_dm_v2.tgz
$ mv cnn_cln data/cnndm
```

### XSUM
```bash
$ wget https://cdn-datasets.huggingface.co/summarization/xsum.tar.gz
$ tar -xzvf xsum.tar.gz
$ mv xsum data/xsum
```

### Generate Augmented Dataset
```bash
$ python generate_augmentation.py \
    --dataset xsum \
    --n 5 \
    --augmentation1 randomdelete \
    --augmentation2 randomswap
```

## Training
### CNN/DM
Our model is warmed up using ```sshleifer/distilbart-cnn-12-6```:
```bash
$ DATA_DIR=./data/cnndm-augmented/RandominsertionRandominsertion-NumSent-3
$ OUTPUT_DIR=./log/cnndm

$ python -m torch.distributed.launch --nproc_per_node=3  cl_finetune_trainer.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --learning_rate=5e-7 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --do_train --do_eval \
  --evaluation_strategy steps \
  --freeze_embeds \
  --save_total_limit 10 \
  --save_steps 1000 \
  --logging_steps 1000 \
  --num_train_epochs 5 \
  --model_name_or_path sshleifer/distilbart-cnn-12-6 \
  --alpha 0.2 \
  --temperature 0.5 \
  --freeze_encoder_layer 6 \
  --prediction_loss_only \
  --fp16
```

### XSUM
```bash
$ DATA_DIR=./data/xsum-augmented/RandomdeleteRandomswap-NumSent-3
$ OUTPUT_DIR=./log/xsum

$ python -m torch.distributed.launch --nproc_per_node=3  cl_finetune_trainer.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --learning_rate=5e-7 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --do_train --do_eval \
  --evaluation_strategy steps \
  --freeze_embeds \
  --save_total_limit 10 \
  --save_steps 1000 \
  --logging_steps 1000 \
  --num_train_epochs 5 \
  --model_name_or_path sshleifer/distilbart-xsum-12-6 \
  --alpha 0.2 \
  --temperature 0.5 \
  --freeze_encoder \
  --prediction_loss_only \
  --fp16
```

## Evaluation
We have released the following checkpoints for pre-trained models as described in the paper:
- [CNN/DM](https://drive.google.com/file/d/1MbLySs5hcxPsSRfPUCzR4AikhtEP08NJ/view?usp=sharing):
- [XSUM](https://drive.google.com/file/d/1SsA8Bstn-VBiH3gDHxU_myFBkNUpNx-D/view?usp=sharing):

### CNN/DM
CNN/DM requires an extra postprocessing step.
```bash
$ export DATA=cnndm
$ export DATA_DIR=data/$DATA
$ export CHECKPOINT_DIR=./log/$DATA
$ export OUTPUT_DIR=output/$DATA

$ python -m torch.distributed.launch --nproc_per_node=2  run_distributed_eval.py \
    --model_name sshleifer/distilbart-cnn-12-6  \
    --save_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --bs 16 \
    --fp16 \
    --use_checkpoint \
    --checkpoint_path $CHECKPOINT_DIR
    
$ python postprocess_cnndm.py \
    --src_file $OUTPUT_DIR/test_generations.txt \
    --tgt_file $DATA_DIR/test.target
```

### XSUM
```bash
$ export DATA=xsum
$ export DATA_DIR=data/$DATA
$ export CHECKPOINT_DIR=./log/$DATA
$ export OUTPUT_DIR=output/$DATA

$ python -m torch.distributed.launch --nproc_per_node=3  run_distributed_eval.py \
    --model_name sshleifer/distilbart-xsum-12-6  \
    --save_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --bs 16 \
    --fp16 \
    --use_checkpoint \
    --checkpoint_path $CHECKPOINT_DIR
```