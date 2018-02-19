#==========================================================================================
# Name: parse_spacy.py
# Description: 
# Author: Maartens Lourens, 2018
#===========================================================================================

import glob
import argparse
import spacy

nlp = spacy.load('en')

parser = argparse.ArgumentParser(description='Syntax Char Rnn pos parsing')
parser.add_argument('--input_dir', '-i', type=str, default='data/shakespeare')
parser.add_argument('--output_dir', '-o', type=str, default='data/shakespeare')
args = parser.parse_args()

# Input files
original_text_files = glob.glob(args.input_dir + "/original*.txt")
syntax_file = args.input_dir + "/text.stx"
text_file = args.input_dir + "/text.txt"

# Read the original text input
parsed_doc = {}
for file in original_text_files:
    print(file)
    with open(file, 'r') as f:
        text = ' '.join(f.readlines()).decode('utf-8')
        # Parse the text
        parsed_doc[file] = nlp(text)

# Concatenate pos tokens and text
new_text = ""
token_text = ""
for file in sorted(parsed_doc.keys()):
    print(file)
    token_text += u''.join([ token.text + " " + token.pos_ + "++" + token.tag_ + "\n"
                   for token in parsed_doc[file] 
                   if token.text != '' and token.pos_.find('SPACE') < 0 ]).encode('utf-8')
    new_text += u''.join([ token.text + " "
                 for token in parsed_doc[file]
                 if token.text != '' and token.pos_.find('SPACE') < 0 ]).encode('utf-8')

# Write syntax file
with open(syntax_file, 'w') as f:
    f.write(token_text)

# Write text file
with open(text_file, 'w') as f:
     f.write(new_text)
