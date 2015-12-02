# -*- coding: utf-8 -*-

import re
import itertools
import random
import nltk
from unidecode import unidecode

# Special tokens modeled after my #sick gamertag
START = 'XxSTARTxX'
STOP = 'XxSTOPxX'

# Maximum bag of words overlap ratio between two sentences
MAX_BOW_OVERLAP = 0.5

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


class Model(object):

  def __init__(self, sentences, token_size=2):
    '''
    sentences: Cleaned and separated sentences from corpus.
    token_size: Number of words in the model's context window.
    '''

    sentences = [self.pad(s, token_size) for s in sentences]

    self.token_size = token_size
    self.dictionary = self.construct(sentences)

  def pad(self, sentence, token_size):
    '''
    sentence: A sentence split out from the corpus.

    Adds padding proportional to token_size.
    '''

    # Pad beginning and end with special indicator tokens
    return [START] * token_size + sentence + [STOP]

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

    # Random pick from weighted dictionary
    # stackoverflow.com/questions/2570690
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

      # Relatively naive sentence splitting on punctuation
      # sentences = self.generate_sentences(text)
      # Works better but sacrifices more of the training data
      sentences = self.generate_sentences_by_char(text)
      print 'Extracted {0} valid sentences from corpus.'.format(len(sentences))

      # Add POS tags to training sentences
      # NB: This takes f*cking forever
      # tagged = []
      # for s in sentences:
        # tagged.append(self.pos_tag(s))

      # Save the training sentences for 'creativity' test
      self.training_words = map(self.remove_padding, sentences)

      self.model = Model(sentences, token_size)

  def pos_tag(self, words):
    '''
    words: Sentence in list form.

    Uses nltk to add part of speech tags to each word in a provided list. 
    '''

    tagged = nltk.pos_tag(words)
    combined = ['__'.join(w) for w in tagged]
    return combined

  def remove_pos_tag(self, words):
    '''
    words: Sentence in list form.

    Removes part of speech suffix tags from each word.
    '''

    def remove(w):
      i = w.find('__')
      if i == -1: return w
      return w[:i]

    words = [remove(w) for w in words]
    return words

  def clean_punctuation(self, sentence):
    '''
    sentence: Sentence split out from the corpus.

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

          # Handle string of single letter abbreviations
          elif sentence[i-2] == '.':
            continue

          # Otherwise, it should end a sentence
          else:
            indexes.append(i)

      for i in range(1, len(indexes)):
        indexes[i] += 1

      # Turn a series of break indexes into tuples that correspond to the range
      # of each complete sentence
      # stackoverflow.com/q/23507320
      slices = zip(indexes, indexes[1::])
      for s in slices:
        separated = sentence[s[0]:s[1]]
        separated = separated.strip()
        split = separated.split(' ')
        # padded = self.split_and_pad(split, self.token_size)
        sentences.append(split)

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
    # print sentences
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

        # handles these cases, more can be added with similar method
        # one issue now is that sentences still can start with "com" or something, making it nonsense
        if (i < len(sentences) - 1) and (sentences[i+1].lower().startswith('com') or sentences[i-1].lower().endswith('inc') or \
          sentences[i+1].lower().startswith('it') or sentences[i+1].lower().startswith('net') or sentences[i+1].lower().startswith('dm')):
          c = sentences[i-1] + sentences[i] + sentences[i+1]
        else:
          c = sentences[i-1] + sentences[i]
        combined.append(c)
    
    # Remove sentences with difficult long-term dependency punctuation
    cleaned = []
    for s in combined:
      s = self.clean_punctuation(s)
      if s != None:
        # Split into array and add padding
        s = s.split(' ')
        cleaned.append(s)
    return cleaned

  def remove_padding(self, words):
    '''
    words: Sentence in list form.

    Removes start and stop indicator tokens.
    '''

    return filter(lambda w: w != START and w != STOP, words)

  def sentence_overlap(self, w1, w2):
    '''
    w1, w2: Word lists with which to compare overlap.

    Returns ratio of language overlap between two sentences.
    '''

    if w1 == None or w2 == None or len(set(w1) | set(w2)) == 0: return 0.0

    # stackoverflow.com/q/29929074
    return len(set(w1) & set(w2)) / float(len(set(w1) | set(w2)))

  def test_sentence(self, words):
    '''
    words: Sentence in list form generated by the Markov model.

    Rejects a sentence that fails to pass tests.
    '''

    # 1) Use a bag of words approach to ensure a sentence has a below
    #    threshold language overlap with any given training sentence
    for w in self.training_words:
      if self.sentence_overlap(words, w) > MAX_BOW_OVERLAP:
        return False

    return True

  def create_sentence(self):
    '''
    Attempts to create a test-passing sentence within a certain number of tries.
    '''

    for i in range(5):
      words = self.model.create_sentence()
      if self.test_sentence(words):
        words = self.remove_pos_tag(words)
        return ' '.join(words)
    return None

