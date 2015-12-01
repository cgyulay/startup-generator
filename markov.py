# -*- coding: utf-8 -*-

import re
import itertools
import random
import nltk
from unidecode import unidecode

# Special tokens modeled after my #sick gamertag
START = 'XxSTARTxX'
STOP = 'XxSTOPxX'

##
# IDEAS
##
# how to improve following sentence coherence
# 1)  use word embeddings to generate a sentence vector
# 2a) adjust markovian word probabilities to maximize cosine similarity
#     between first sentence and following in-progress sentence
# 2b) generate many following sentences and pick highest given cosine
#     similarity between first and following complete sentence
#
# 1)  use bag of words to ensure similar language among sentences
# 2)  combine BOW with embeddings?
##
# how to improve sentence separation/filtration
# 1)  assume sentences break on period/exclamation/question
# 2)  if sentence is below some threshhold length, assume it's abbrevs
# 3)  if possible filter out sentences with problematic long-term
#     dependencies (quotes, parens, brackets)
##


class Model(object):

  def __init__(self, sentences, token_size=2):
    '''
    sentences: Cleaned and separated sentences from corpus.
    token_size: Number of words in the model's context window.
    '''

    self.token_size = token_size
    self.dictionary = self.construct(sentences)

  def construct(self, sentences):
    '''
    Constructs a nested dictionary that stores all preceding states of length
    token_size, and the transitions from each state with their associated
    probabilities.
    '''

    dictionary = {}

    for sentence in sentences:
      for i in range(len(sentence) - self.token_size):
        preceding = tuple(sentence[i:i + self.token_size])
        following = sentence[i + self.token_size]

        if preceding not in dictionary:
          dictionary[preceding] = {}

        if following not in dictionary[preceding]:
          dictionary[preceding][following] = 1
        else:
          dictionary[preceding][following] += 1
    return dictionary

  def next(self, preceding):
    '''
    Picks the next word in the chain by sampling from the weighted distribution
    of the preceding state.
    '''

    d = self.dictionary[preceding]

    # Random pick from weighted dictionary found here
    # http://stackoverflow.com/questions/2570690
    total = sum(d.itervalues())
    pick = random.randint(0, total - 1)
    acc = 0

    # Accumulate weights until the random pick is in range
    for key, weight in d.iteritems():
      acc += weight
      if pick < acc:
        return key

  def create_sentence(self):
    '''
    Creates a sentence by probabilistically choosing the next word given a
    preceding state, until receiving a stop token.
    '''

    # By default, start from a random beginning
    preceding = (START,) * self.token_size
    traversing = True
    words = []

    while traversing:
      following = self.next(preceding)
      traversing = following != STOP
      if traversing:
        words.append(following)
        preceding = preceding[1:] + (following,)
    return words


class Generator(object):

  def __init__(self, corpus_path, token_size=2):
    '''
    corpus_path: Path to training corpus.
    token_size: Number of words in the model's context window.

    Takes a corpus and produces a queryable model that generates sentences.
    '''

    self.token_size = token_size

    with open(corpus_path) as f:
      text = f.read()
      # sentences = self.generate_sentences(text)
      sentences = self.generate_sentences_by_char(text)
      # self.model = Model(sentences, token_size)

  def clean_punctuation(self, sentence):
    '''
    sentence: A sentence split out from the corpus.

    Rejects sentences with troublesome long-term dependency punctuation, then
    replaces non-standard spaces.
    '''

    # For simplicity, discard sentences with ? or ! (~.5% training set)
    if not any(itertools.imap(sentence.__contains__, '"?!()[]‘’“”')):
      # Remove non-standard spaces
      sentence = sentence.decode('utf8')
      sentence = sentence.replace(u'\xa0', u' ')
      return unidecode(sentence)
    else: return None

  def split_and_pad(self, sentence, token_size):
    '''
    sentence: A sentence split out from the corpus.

    Converts a sentence into an array and adds padding proportional to
    token_size.
    '''

    separated = sentence.split(' ')

    # Pad beginning and end with special indicator tokens
    return [START] * token_size + separated + [STOP]

  def generate_sentences_by_char(self, text):
    '''
    text: Text from training corpus.

    Examines the corpus character by character, identifies locations of
    contextually probable sentence breaks, and divides into sentences.
    '''

    # First, outright discard entries with difficult punctuation
    cleaned = [self.clean_punctuation(s) for s in text.split('\n')]
    cleaned = [s for s in cleaned if s != None]

    sentences = []
    for sentence in cleaned:
      indexes = [0]
      for i in range(len(sentence)):
        if sentence[i] == '.':

          # Handle Inc.
          if sentence[i-3:i].lower() == 'inc':
            continue

          # Handle last char (short-circuiting)
          elif i == len(sentence)-1:
            indexes.append(i)

          # Handle tlds (.com, .net, etc)
          elif sentence[i+1] != ' ':
            continue

          # Otherwise, it should end a sentence
          else:
            indexes.append(i)

      for i in range(len(indexes)):
        if indexes[i] == len(sentence)-1: break
        split = sentence[indexes[i]:indexes[i+1]]
        sentences.append(split)


    # This is very close to being done, just need to work out last spacing
    # kinks and convert to list of lists  
    return sentences

  def generate_sentences(self, text):
    '''
    text: Text from training corpus.

    Divides the corpus into sentences naively, using punctuation indicators
    to split sentences.
    '''

    # Break sentences on punctuation
    sentences = re.split('(!|\?|\.|\n)', text)
    sentences = [s for s in sentences if s != '\n' and s != '']
    sentences = [s.strip() for s in sentences]

    skip_next = False
    combined = []
    for i in range(1, len(sentences)):
      if skip_next:
        skip_next = False
        continue

      if sentences[i] in ['.', '!', '?']:
        # TODO
        # Need to specifically handle tlds like .com, .net for this dataset
        # Assume there are no sentences with one word
        # Can pass tuple to startswith if helpful
        # if sentences[i+1].startswith('com '):

        c = sentences[i-1] + sentences[i]
        combined.append(c)
    
    # Remove sentences with difficult long-term dependency punctuation
    cleaned = []
    for s in combined:
      s = self.clean_punctuation(s)
      if s != None:
        # Split into array and add padding
        padded = self.split_and_pad(s, self.token_size)
        cleaned.append(padded)
    return cleaned

  def create_sentence(self):
    words = self.model.create_sentence()
    return ' '.join(words)
