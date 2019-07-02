
import argparse

import numpy as np
import os
from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = 128  # batch size for each GPU
    #使用GPU的数量
    n_gpus = 2
    #设置在哪两个GPU上运行，它是并行的
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    #训练语料中的词数，是不去重的，这会影响到训练的时间，需要结合自己的训练语料修改
    n_train_tokens = 768648884

    options = {
     'bidirectional': True,
     #中文的去掉
     # 'char_cnn': {'activation': 'relu',
     #  'embedding': {'dim': 16},
     #  'filters': [[1, 32],
     #   [2, 32],
     #   [3, 64],
     #   [4, 128],
     #   [5, 256],
     #   [6, 512],
     #   [7, 1024]],
     #  'max_characters_per_token': 50,
     #  'n_characters': 261,
     #  'n_highway': 2},
    
     'dropout': 0.1,
     #设置的LSTM的参数，可以修改
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 512,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
    #一个批次负采样的个数，语料过短时需要修改,修改小点
     'n_negative_samples_batch': 20,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #检测点等一些文件的输出路径，这个文件夹提前创建好
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    #根据训练语料制定的词表的路径
    parser.add_argument('--vocab_file', help='Vocabulary file')
    #训练语料的路径
    parser.add_argument('--train_prefix', help='Prefix for train files')

    args = parser.parse_args()
    main(args)

