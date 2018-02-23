from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch

import numpy as np
import math
import os
import argparse
import pickle

from rnn import *
import sys

parser = argparse.ArgumentParser(description='Syntax Char Rnn Pre-Processing')
parser.add_argument('--seq_length', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--rnn_size', type=int, default=128)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=2e-3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--seed', type=str, default='a')
parser.add_argument('--input_dir', '-i', type=str, default='data/austen')
parser.add_argument('--output_dir', '-o', type=str, default='data/austen')
args = parser.parse_args()

# CUDA
use_cuda = torch.cuda.is_available()

# randomise runs
torch.manual_seed(np.random.randint(1,9999))
random_state = np.random.RandomState(np.random.randint(1,9999))

seq_length = args.seq_length
batch_size = args.batch_size
hidden_size = args.rnn_size
epoch_count = args.max_epochs
n_layers = args.num_layers
lr = args.learning_rate
dropout = args.dropout
checkpoint_prepend = os.path.join(args.output_dir, 'checkpoint_')
final_checkpoint_prepend = os.path.join(args.output_dir, 'final_checkpoint_')

#
# Load pre-processed files of data structures
#

# text
text_file = args.input_dir + "/text.pkl"
with open(text_file, 'rb') as f:
    text = pickle.load(f)

# base_text_dict
base_text_dict_file = args.input_dir + "/text_base_text_dict.pkl"
with open(base_text_dict_file, 'rb') as f:
    base_text_dict = pickle.load(f)

# combos_text_num
text_combos_text_num_file = args.input_dir + "/text_combos_text_num.pkl"
with open(text_combos_text_num_file, 'rb') as f:
    combos_text_num = pickle.load(f)

# combos
text_combos_file = args.input_dir + "/text_combos.pkl"
with open(text_combos_file, 'rb') as f:
    combos = pickle.load(f)

## combo_to_index
#combo_to_index_file = args.input_dir + '/text_combo_to_index.pkl'
#with open(combo_to_index_file, 'rb') as f:
#    combo_to_index = pickle.load(f)

# index_to_char
index_to_char_file = args.input_dir + '/text_index_to_char.pkl'
with open(index_to_char_file, 'rb') as f:
    index_to_char = pickle.load(f)

# combos: the final list of ordered numbers used to operate on
combos = sorted(set(combos_text_num))
combos_len = len(combos)

def chunks(l, n):
    for i in range(0, len(l) - n, n):
        yield l[i:i + n]

def train():
    # convert all combos to indices
    #batches = [combo_to_index[text_num] for text_num in combos_text_num]
    batches = [text_num for text_num in combos_text_num]

    # chunk into sequences of length seq_length + 1
    batches = list(chunks(batches, seq_length + 1))

    # chunk sequences into batches
    batches = list(chunks(batches, batch_size))

    # convert batches to tensors and transpose
    # each batch is (sequence_length + 1) x batch_size
    batches = [torch.LongTensor(batch).transpose_(0, 1) for batch in batches]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    hidden = Variable(model.create_hidden(batch_size))

    if use_cuda:
        hidden = hidden.cuda()
        model.cuda()

    all_losses = []

    try:
        epoch_progress = tqdm(range(1, epoch_count + 1))
        best_loss = float('inf')
        for epoch in epoch_progress:
            random_state.shuffle(batches)

            batches_progress = tqdm(batches)
            for batch, batch_tensor in enumerate(batches_progress):
                if use_cuda:
                    batch_tensor = batch_tensor.cuda()

                # reset the model
                model.zero_grad()

                # everything except the last
                input_variable = Variable(batch_tensor[:-1])

                # everything except the first, flattened
                # what does this .contiguous() do?
                target_variable = Variable(batch_tensor[1:].contiguous().view(-1))

                # prediction and calculate loss
                output, _ = model(input_variable, hidden)
                loss = loss_function(output, target_variable)

                # backprop and optimize
                loss.backward()
                optimizer.step()

                loss = loss.data[0]
                best_loss = min(best_loss, loss)
                all_losses.append(loss)

                batches_progress.set_postfix(loss='{:.03f}'.format(loss))
                if loss < 1.3 and loss == best_loss:
                    checkpoint_path = os.path.join(args.output_dir, 'checkpoint_tl_')
                    checkpoint_path = checkpoint_path + str('{:.03f}'.format(loss)) + '.cp'
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, checkpoint_path)
                if loss < 0:
                    exit()
   
            epoch_progress.set_postfix(loss='{:.03f}'.format(best_loss))
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_ep_')
            checkpoint_path = checkpoint_path + str('{:.03f}'.format(loss)) + '.cp'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)

    except KeyboardInterrupt:
        pass

    # final save
    final_path = os.path.join(args.output_dir, 'final_checkpoint_')
    final_path = final_path + str('{:.03f}'.format(loss)) + '.cp'
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, final_path)

model = RNN(combos_len, hidden_size, combos_len, n_layers, dropout)
train()
