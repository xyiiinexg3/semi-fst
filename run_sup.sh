#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python T5_Large_filter.py \
    -style 0 \
    -dataset em \
    -order em.sup \
    -batch_size 8 \
    -val_batch_size 16 \
    -lr 2e-5

# python T5_Large_filter.py -style 0 -dataset em  -order em.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5
# python T5_Large_filter.py -style 0 -dataset fr  -order fr.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5 

em
1
# python T5_Large_filter.py -style 0 -dataset em  -order em.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5
# python T5_Large_filter.py -style 0 -dataset em  -order em.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5  --warmup False  -isReload
2; pre-step = 1 epoch 
# python T5_Large_filter.py -style 0 -dataset em  -order em.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5
3 新的spell aug(warmup 怎么 占比呢-》 全部 + spell contrastive loss) warmup only + cont-spell
python T5_Large_filter.py -style 0 -dataset em  -order em.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5  --warmup False  -isReload -isReload_opt True
4 warmup only + cont-spell-gec
python T5_Large_filter.py -style 0 -dataset em  -order em.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5  --warmup False  -isReload 
5 warmup only + cont-spell-gec-src
python T5_Large_filter.py -style 0 -dataset em  -order em.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5  --warmup False  -isReload 

fr
2; pre = 5 epoch + cont
# python T5_Large_filter.py -style 0 -dataset fr  -order fr.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5
1 warmup only + cont
# python T5_Large_filter.py -style 0 -dataset fr  -order fr.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5
# python T5_Large_filter.py -style 0 -dataset fr  -order fr.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5  --warmup False  -isReload # -isReload_opt True # 断电
3 
python T5_Large_filter.py -style 0 -dataset fr  -order fr.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5  --warmup False  -isReload # 在warmup only的基础上继续跑
4 warmup only + cont-spell-gec
python T5_Large_filter.py -style 0 -dataset fr  -order fr.sup  -batch_size 8  -val_batch_size 16  -lr 2e-5  --warmup False  -isReload 
