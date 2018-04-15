# Syntax Char RNN

## Introduction

Char RNN is great for creative language projects, but also limited. Syntax Char RNN attempts to expand traditional Char RNN capabilities by encoding context in the form of syntactic Parts of Speech.

The usefulness of Syntax Char RNN over pure Char RNN, at least in the present version, lies in texts where the parser is fairly accurate and the complexity high. [spaCy](http://spacy.io) is used as the default parser due to its speed and useability. Instructions for using the DRAGNN parser are also provided for comparison and experimentation. 

Please see the companion [blog post](https://thecombedthunderclap.blogspot.com/2018/02/syntax-char-rnn-for-context-encoding.html) for a broader discussion.

## Installation

Make sure you have a recent version of Python 2.7

```
git clone https://github.com/thundercomb/pytorch-syntax-char-rnn
cd pytorch-syntax-char-rnn
pip install spacy numpy tqdm torch 
python -m spacy download en
```

## Input

Provide a text file(s) as ```original*.txt``` where ```*``` is expandable if there is to be more than one file, or empty if not. Eg.: 

```
data/austen/original.txt
```
or
```
data/austen/original1.txt
data/austen/original2.txt
data/austen/original3.txt
```

If the file is too large for spaCy to process, split it into separate files at meaningful break points like paragraphs or chapters.

## Run

### Parse

The parse program takes two arguments:  
```--input_dir``` : the directory where the input file ```original*.txt``` resides  
```--output_dir``` : the output directory (typically the same, but can be different)  

The parse program outputs two files:  
```text.txt``` : a tokenised version of ```original*.txt```  
```text.stx``` : syntactic Parts of Speech for each of the text tokens  

```
python2.7 parse.py --input_dir data/austen/ --output_dir data/austen/
```

### Prepare

The prepare program reads the parsed files, matches the text tokens with parts of speech, calculates indexes, and saves the data structures as files for use in the training process.

The prepare program takes two arguments:  
```--input_dir``` : the directory where the parsed files ```text.txt``` and ```text.stx``` reside  
```--output_dir``` : the output directory (typically the same, but can be different)  

The prepare program outputs two files:  
```text.pkl``` : lightly processed version of ```text.txt```  
```text_base_text_dict.pkl``` : dictionary of lists with matching text tokens and pos types  
```text_combos_text_num.pkl``` : list of encoded composite units (char indexes and pos type indexes)  
```text_index_to_char.pkl``` : dictionary of char indexes and accompanying chars  
```text_combos.pkl``` : sorted list of unique composite units  

```
python2.7 prepare.py --input_dir data/austen --output_dir data/austen
```

### Train

The train program reads the data structure files and trains the neural network on chunks of the composite units encoded during the prepare process.

The train program takes ten arguments:  
```--seq_length``` : sequence length (default=50)  
```--batch_size``` : minibatch size (default=50)  
```--rnn_size``` : hidden state size (default=128)  
```--num_layers``` : number of rnn layers (default=2)  
```--max_epochs``` : maximum number of epochs (default=10)  
```--learning_rate``` : learning rate (default=2e-3)  
```--dropout``` : dropout (default=0.5)  
```--seed``` : seeding character(s) (default='a')  
```--input_dir``` : input directory (default='data/austen')  
```--output_dir``` : output directory (default='data/austen')  

The train program outputs checkpoints after each epoch, or whenever a better training loss under 1.3 is achieved.

```
python2.7 train.py --input_dir data/austen/ --output_dir data/austen/ --max_epochs 200 --seq_length 135 --rnn_size 256
```

### Generate

The generate program samples a model loaded from a specified checkpoint and prints the results.

The generate program takes five arguments:  
```--temperature``` : number between 0 and 1; lower means more conservative predictions for sampling (default=0.8)  
```--sample_len``` : number of characters to sample (default=500)  
```--checkpoint``` : checkpoint model to load  
```--input_dir``` : input directory containing from which to load data (default='data/austen')  

The generate program outputs sampled text to the terminal.

```
python2.7 generate.py --input_dir data/austen/ --checkpoint data/austen/checkpoint_0.708.cp --sample_len 2000 --temperature 1
```

## TODO

* Reduce the number of redundant space encodings  
* Calculate a more granular weighting for type  
* Investigate further candidates for context encoding, over and above syntax  
* Investigate more elegant ways of grouping encoding contexts  
* Add validation loss for comparison to training loss  
* Estimate parser accuracy for a specific text  
* Add tests  
* ~~Run on GPUs!~~

## Contact

Feel free to comment, raise issues, or create pull requests. You can also reach out to me on Twitter [@thundercomb](https://twitter.com/thundercomb). 

## License

The software is licensed under the terms of the [GNU Public License v2](http://github.com/thundercomb/poetrydb/LICENSE.txt). It basically means you can reuse and modify this software as you please, as long as the resulting program(s) remain open and licensed in the same way.
