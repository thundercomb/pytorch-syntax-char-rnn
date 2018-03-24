from torch.autograd import Variable
import torch.nn as nn
import torch

import numpy as np
import os
import re
import pickle
import argparse

from rnn import *

parser = argparse.ArgumentParser(description='PyTorch syntax-combo-rnn')
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--sample_len', type=int, default=500)
parser.add_argument('--checkpoint', '-c', type=str)
parser.add_argument('--seed', type=str, default=12)
parser.add_argument('--input_dir', '-i', type=str, default='data/austen')
args = parser.parse_args()

combos_file = args.input_dir + "/text_combos.pkl"
with open(combos_file, 'rb') as f:
    combos = pickle.load(f)

index_to_char_file = args.input_dir + "text_index_to_char.pkl"
with open(index_to_char_file, 'rb') as f:
    index_to_char = pickle.load(f)

combos = sorted(list(set(combos)))
combos_len = len(combos)

random_state = np.random.RandomState(np.random.randint(1,9999))

def uppercase_sentences(match):
    return match.group(1) + ' ' + match.group(2).upper()

def index_to_tensor(index):
    tensor = torch.zeros(1, 1).long()
    tensor[0,0] = index
    return Variable(tensor)

def manual_sample(x, temperature):
    x = x.reshape(-1).astype(np.float)
    x /= temperature
    x = np.exp(x)
    x /= np.sum(x)
    x = random_state.multinomial(1, x)
    x = np.argmax(x)
    return x.astype(np.int64)

def sample(model, prime_combo, predict_len, temperature):
    hidden = Variable(model.create_hidden(1), volatile=True)
    prime_tensors = [index_to_tensor(prime_combo)]

    for prime_tensor in prime_tensors[-2:]:
        _, hidden = model(prime_tensor, hidden)

    inp = prime_tensors[-1]
    predicted = [ prime_combo ]
    for p in range(predict_len):
        output, hidden = model(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Alternative: use numpy
        # top_i = manual_sample(output.data.numpy(), temperature)

        # Add predicted character to string and use as next input
        # Type indexes start after character indexes; we only want chars
        predicted_combo = top_i
        if predicted_combo < len(index_to_char): 
            predicted = predicted + [ predicted_combo ]
        inp = index_to_tensor(predicted_combo)

    return predicted

if os.path.exists(args.checkpoint):
    print('Parameters found at {}... loading'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
else:
    raise ValueError('File not found: {}'.format(args.checkpoint))

hidden_size = checkpoint['model']['encoder.weight'].size()[1]
n_layers = 0
for key in checkpoint['model'].keys():
    if 'cells.weight_hh' in key:
        n_layers = n_layers + 1

model = RNN(combos_len, hidden_size, combos_len, n_layers, 0.5)
model.load_state_dict(checkpoint['model'])
sample = sample(model, args.seed, args.sample_len, args.temperature)

new_text = ""
for combo in sample[1:]:
    new_text = new_text + index_to_char[combo]
    
new_text = new_text.split(' ', 1)[1].capitalize()
new_text = re.sub(r'([.?!]) ([a-z])', uppercase_sentences, new_text)
new_text = re.sub(r'([.?!]\n)([a-z])', uppercase_sentences, new_text)
new_text = re.sub(r'([.?!]\n *\n)([a-z])', uppercase_sentences, new_text)
if new_text.find('.') and new_text[:new_text.rfind('.')+1] != '':
    new_text = new_text[:new_text.rfind('.')+1]

print(new_text)
