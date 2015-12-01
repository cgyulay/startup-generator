# -*- coding: utf-8 -*-

import re
import itertools
import random
import nltk
from unidecode import unidecode

# Special tokens modeled after my #sick gamertag
START = 'XxSTARTxX'
STOP = 'XxSTOPxX'

DEFAULT_MAX_OVERLAP_RATIO = 0.7
DEFAULT_MAX_OVERLAP_TOTAL = 15
DEFAULT_TRIES = 10

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
    '''

    self.token_size = token_size

    with open(corpus_path) as f:
      text = f.read()
      sentences = list(self.generate_sentences(text))
      self.model = Model(sentences, token_size)

  def generate_sentences(self, text):
    '''
    text: Text from training corpus.
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
      if not any(itertools.imap(s.__contains__, '"()[]‘’“”')):

        # Remove non-standard spaces
        s = s.decode('utf8')
        s = s.replace(u'\xa0', u' ')
        # s = s.encode('ascii')
        s = unidecode(s)

        # Convert to array of words separated by spaces
        separated = s.split(' ')

        # Pad beginning and end with special indicator tokens
        padded = [START] * self.token_size + separated + [STOP]
        cleaned.append(padded)


    return cleaned

  def create_sentence(self):
    words = self.model.create_sentence()
    return ' '.join(words)
