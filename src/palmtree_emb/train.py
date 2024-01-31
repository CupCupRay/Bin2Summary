import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from palmtree_utils.config import *
import numpy as np
import palmtree
from palmtree import model
from palmtree import dataset
from palmtree import trainer
import pickle as pkl
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

print(palmtree.__file__)

data_path = '../../data/'

if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--Dataset', '-d', required=False, help='Input the "train" or "test" set you need.')
    args = parser.parse_args()
    my_choice = args.Dataset

    if my_choice.lower() == 'train':
        vocab_path = data_path + "train_vocab"
        train_cfg_dataset = data_path + "cfg_train_pair.txt"
        train_dfg_dataset = data_path + "dfg_train.txt"
        test_dataset = data_path + "cfg_train_single.txt"
        output_path = data_path + "train_transformer"
        workers = 10
    else:
        vocab_path = data_path + "test_vocab"
        train_cfg_dataset = data_path + "cfg_test_pair.txt"
        train_dfg_dataset = data_path + "dfg_test.txt"
        test_dataset = data_path + "cfg_test_single.txt"
        output_path = data_path + "test_transformer"
        workers = 0

    with open(train_cfg_dataset, "r", encoding="utf-8") as f1:
        with open(train_dfg_dataset, "r", encoding="utf-8") as f2:
            vocab = dataset.WordVocab([f1, f2], max_size=13000, min_freq=1)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(vocab_path)

    print("Loading Vocab", vocab_path)
    vocab = dataset.WordVocab.load_vocab(vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", train_cfg_dataset)
    print("Loading Train Dataset", train_dfg_dataset)
    train_dataset = dataset.BERTDataset(train_cfg_dataset, train_dfg_dataset, vocab, seq_len=20,
                                        corpus_lines=None, on_memory=True)

    print("Loading Test Dataset", test_dataset)
    test_dataset = dataset.BERTDataset(test_dataset, test_dataset, vocab, seq_len=20, on_memory=True) \
        if test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=256, num_workers=workers)

    test_data_loader = DataLoader(test_dataset, batch_size=256, num_workers=workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = model.BERT(len(vocab), hidden=128, n_layers=12, attn_heads=8, dropout=0.0)

    print("Creating BERT Trainer")
    trainer = trainer.BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader,
                                  test_dataloader=test_data_loader,
                                  lr=1e-5, betas=(0.9, 0.999), weight_decay=0.0,
                                  with_cuda=True, cuda_devices=[0, 1], log_freq=100)

    print("Training Start")
    for epoch in range(20):
        trainer.train(epoch)
        trainer.save(epoch, output_path)
