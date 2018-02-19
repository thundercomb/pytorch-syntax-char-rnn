#==========================================================================================
# Name: prepare.py
# Description: Read in (1) a text file and (2) a syntax parsed file with POS data about the 
#              text file. The data in the syntax parsed file may or not be in the correct
#              order, depending on the syntax parser used.
#              The data is read into dictionary arrays, and the correct order of syntax data
#              calculated.
#              From this a definitive set of encodings and indexes are constructed 
#              and saved for later use by the training script.
# Author: Maartens Lourens, 2018
#===========================================================================================

from tqdm import tqdm

import numpy as np
import math
import os
import argparse
import pickle
import itertools
import sys

parser = argparse.ArgumentParser(description='Syntax Char Rnn Pre-Processing')
parser.add_argument('--input_dir', '-i', type=str, default='data/shakespeare')
parser.add_argument('--output_dir', '-o', type=str, default='data/shakespeare')
args = parser.parse_args()

# Input files
input_syntax_file = args.input_dir + "/text.stx"
input_text_file = args.input_dir + "/text.txt"

# Create text array in base dict
# Read formatted text into string
base_text_dict = {}
base_text_dict['data'] = []
base_text_dict['type'] = []
with open(input_text_file, 'r') as f:
    for line in f:
        parts = line.split()
        base_text_dict['data'].extend(parts)
text = ' '.join(base_text_dict['data'])

# Save text file for re-use in training process
text_pickle_file = os.path.splitext(input_text_file)[0] + '.pkl'
with open(text_pickle_file, 'wb') as f:
    pickle.dump(text, f)

# Create char to index and index to char dictionaries
chars = sorted(list(set(text)))
chars_len = len(chars)
char_to_index = {}
index_to_char = {}
for i, c in enumerate(chars):
    char_to_index[c] = i
    index_to_char[i] = c

# Read the parsed syntax input into a dictionary with two arrays
# One array for syntax types, the other for the text data
proc_text_dict = {}
proc_text_dict['data'] = []
proc_text_dict['type'] = []
with open(input_syntax_file, 'r') as f:
    for line in f:
        row = line.split()
        proc_text_dict['data'].append(row[0])
        proc_text_dict['type'].append(row[1])

# Order the processed syntax types according to the base text data
# In the case of DRAGNN and spaCy parsed data the data is already in the correct order,
# but SyntaxNet's output is according to the tree structure, so needs to be reordered
#
# The logic to do so is straightforward and is set out in the below example
# Note that due to the for loops this version takes way too long:
#
# for i, word in enumerate(base_text_dict['data']):
#    for j, pword in enumerate(proc_text_dict['data']):
#        if pword == word:
#            base_text_dict['type'][i] = proc_text_dict['type'][j]
#            proc_text_dict['data'][j] = ''
#            proc_text_dict['type'][j] = ''
#            break
#
# The following version improves performance, but the remaining for loop slows it down:
#
# for i, word in enumerate(base_text_dict['data']):
#     base_text_dict['type'].append(next(([proc_text_dict['type'].pop(j),proc_text_dict['data'].pop(j)]  for j, pword in enumerate(proc_text_dict['data']) if pword == word), 'default')[0])
#
# This version uses an iterator and a generator and solves performance issues.
# Notes: 
# - pop(j) removes the found entries so they're not taken into account more than once
# - next(..) breaks out of the internal loop once an instance is found
#
base_text_dict['type'] = list(
    [proc_text_dict['type'].pop(j),proc_text_dict['data'].pop(j)][0]
    for word in base_text_dict['data']
    for j, pword in
    next([x] for x in enumerate(proc_text_dict['data'])
    if x[1] == word)
    if pword == word)

# Save base text dictionary for good luck
base_text_dict_file = os.path.splitext(input_text_file)[0] + '_base_text_dict.pkl'
with open(base_text_dict_file, 'wb') as f:
    pickle.dump(base_text_dict, f)

# Create dictionary to translate syntax types to index and vice versa
syntax_types = sorted(list(set(base_text_dict['type'])))
syntax_types_len = len(syntax_types)
syntax_type_to_index = {}
index_to_syntax_type = {}
for i, c in enumerate(syntax_types):
    # Keep consistent digits
    syntax_type_to_index[c] = i
    index_to_syntax_type[i] = c

# Read the parsed syntax input into a dictionary with two arrays
# One array for syntax types, the other for the text data

# Leftmost 5 digits are reserved for syntax types (index * 1000)
# The rest (rightmost 3 digits) will be for characters
#
# We use list comprehension to speed things up
# A more readable set of steps would be as follows
#
# for i, word in enumerate(base_text_dict['data']):
#     print(word)
#     for char in word:
#         print(char)
#         word_type = base_text_dict['type'][i]
#         char_num = syntax_type_to_index[word_type] * 1000 + char_to_index[char]
#         combos_text_num = combos_text_num + [char_num]
#         combo_to_char[char_num] = char
#     char_num = 90000000 + char_to_index[' ']
#     combos_text_num = combos_text_num + [char_num]
#     combo_to_char[char_num] = ' '

combos_text_num = []
combo_to_char = {}
cti_length = len(char_to_index)
combos_text_num = [ 
    # the word's type index followed by space index
    #[syntax_type_to_index[base_text_dict['type'][i]] + cti_length,char_to_index[char]] if char == ' '
    [char_to_index[char],syntax_type_to_index[base_text_dict['type'][i]] + cti_length,syntax_type_to_index[base_text_dict['type'][i]] + cti_length] if char == ' '
    # character index first, type index second
    else [char_to_index[char]]
    for i, word in enumerate(base_text_dict['data'])
    for char in ' ' + word ]
# Now flatten the list
combos_text_num = [item for items in combos_text_num for item in items]
# Remove spaces before punctuation and apostrophes
# PS: This is best effort, the POS parser's results complicates things
problem_list = ['PUNCT', 'PART++POS']
to_remove_list = [
    value + cti_length 
    for key, value in syntax_type_to_index.items() 
    for type in problem_list 
    if key.find(type) >= 0 ]
[ combos_text_num.pop(syntax_index) 
    for syntax_index in sorted([
        i for i, pair in enumerate(itertools.izip(combos_text_num, combos_text_num[1:])) 
        for r in to_remove_list if pair == (0, r)], reverse = True)]

# Save combos text num array for re-use in training process
combos_text_num_file = os.path.splitext(input_text_file)[0] + '_combos_text_num.pkl'
with open(combos_text_num_file, 'wb') as f:
    pickle.dump(combos_text_num, f)

# combos: the final list of ordered numbers used to operate on
combos = sorted(set(combos_text_num))

# Save index to char file for re-use in training process
index_to_char_file = os.path.splitext(input_text_file)[0] + '_index_to_char.pkl'
with open(index_to_char_file, 'wb') as f:
    pickle.dump(index_to_char, f)

combos_len = len(combos)

# Save combos to combosfile for re-use in generate process
combofile = os.path.splitext(input_text_file)[0] + '_combos.pkl'
with open(combofile, 'wb') as f:
    pickle.dump(combos, f)
