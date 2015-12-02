# -*- coding: utf-8 -*-

'''
Separates and cleans sentences, then adds part of speech tags and saves to and
new corpus.
'''

import nltk
import itertools
import time
from unidecode import unidecode

class Tagger(object):

  def __init__(self, corpus_path, dest_path):
    with open(corpus_path) as f:
      text = f.read()

      # Separate corpus into sentences
      sentences = self.generate_sentences_by_char(text)
      print 'Extracted {0} valid sentences from corpus.'.format(len(sentences))

      # Add POS tags
      print 'Beggining POS tagging.'
      prev_time = time.time()
      tagged = []

      for i, s in enumerate(sentences):
        tagged.append(self.pos_tag(s))

        if i % 10 == 0:
          elapsed = time.time() - prev_time
          print 'Tagged {0} lines in {1}s.'.format(i, elapsed)
          prev_time = time.time()

      print 'Completed POS tagging.'

      # Save to file
      dest = open(dest_path, 'w')
      for s in tagged:
        dest.write(' '.join(s) + '\n')
      dest.close()
      print 'Finished writing to {0}'.format(dest_path)

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
        sentences.append(split)

    return sentences

# Run
corpus_path = 'corpora/crunchbase_descriptions_micro.txt'
dest_path = 'tagged_' + corpus_path
tagger = Tagger(corpus_path, dest_path)
