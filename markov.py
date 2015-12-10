# -*- coding: utf-8 -*-

import re
import itertools
import random
from unidecode import unidecode

# Special tokens modeled after my #sick gamertag
START = 'XxSTARTxX'
STOP = 'XxSTOPxX'

# Maximum bag of words overlap ratio between two sentences
MAX_BOW_OVERLAP = 0.6

class Model(object):

  def __init__(self, sentences, token_size=2):
    '''
    sentences: Cleaned and separated sentences from corpus.
    token_size: Number of words in the model's context window.

    This is the meat and potatoes of the project. The Model class builds a
    transition model from a list of sentences and can be asked to generate new
    sentences from these transitions.
    '''

    self.token_size = token_size
    sentences = [self.pad(s, token_size) for s in sentences]

    self.dictionary = self.construct(sentences, self.token_size)

  def pad(self, sentence, token_size):
    '''
    sentence: A sentence split out from the corpus.
    token_size: Number of words in the model's context window.

    Adds padding proportional to token_size.
    '''

    # Pad beginning and end with special indicator tokens
    return [START] * token_size + sentence + [STOP]

  def construct(self, sentences, token_size):
    '''
    sentences: A list of cleaned sentences.
    token_size: Number of words in the model's context window.

    Constructs a nested dictionary that stores all preceding states of length
    token_size, and the transitions from each state with their associated
    probabilities.
    '''

    dictionary = {}

    for sentence in sentences:
      for i in range(len(sentence) - token_size):
        preceding = tuple(sentence[i:i + token_size])
        following = sentence[i + token_size]

        if preceding not in dictionary:
          dictionary[preceding] = {}

        if following not in dictionary[preceding]:
          dictionary[preceding][following] = 1
        else:
          dictionary[preceding][following] += 1
    return dictionary

  def next(self, dictionary, preceding):
    '''
    dictionary: A dictionary that stores the state transition model.
    preceding: A tuple representing the state from which to transition.

    Picks the next word in the chain by sampling from the weighted distribution
    of the preceding state.
    '''

    # Cover situations where a longer state doesn't exist
    d = dictionary.get(preceding, None)
    if d == None: return ('', 0)

    # Random pick from weighted dictionary
    # stackoverflow.com/questions/2570690
    total = sum(d.itervalues())
    pick = random.randint(0, total - 1)
    acc = 0

    # Accumulate weights until the random pick is in range
    for key, weight in d.iteritems():
      acc += weight
      if pick < acc:
        # Return both the key and the total available destination states
        next_state = preceding[1:] + (key,)
        total_destinations = len(dictionary.get(next_state, {}))
        return key, total_destinations

  def create_sentence(self, preceding=None):
    '''
    preceding: A tuple of words representing the desired starting state. Must
               be of length < self.token_size.

    Creates a sentence by probabilistically choosing the next word given a
    preceding state, until receiving a stop token.
    '''

    # By default, start from a random beginning
    if preceding == None:
      preceding = (START,) * self.token_size
    elif len(preceding) > self.token_size:
      preceding = preceding[:token_size]
    elif len(preceding) < self.token_size:
      preceding = (START,) * (self.token_size - len(preceding)) + preceding

    traversing = True
    words = []

    while traversing:
      following = self.next(self.dictionary, preceding)[0]
      traversing = following != STOP
      if traversing:
        words.append(following)
        preceding = preceding[1:] + (following,)

    return words


class MultiModel(Model):

  def __init__(self, sentences, token_size=2):
    '''
    MultiModel builds all models with state sizes 1->token_size, rather than
    just a single model with a state length of token_size. This allows a
    dynamic transition model that can rely on any number of preceding words.
    '''

    self.token_size = token_size
    sentences = [self.pad(s, token_size) for s in sentences]

    self.dictionaries = {}
    for i in range(1, self.token_size+1):
      self.dictionaries[i] = self.construct(sentences, i)

  def create_sentence(self):
    '''
    Creates a sentence by probabilistically choosing the next word using
    dictionaries of variable token_size, and then selecting the word based on
    the dictionary with the ideal number of destination states.
    '''

    preceding = (START,) * self.token_size
    traversing = True
    words = []

    while traversing:
      destinations = []
      for d in self.dictionaries:
        following, dest_count = self.next(self.dictionaries[d], preceding[-d:])
        if following != START: destinations.append((dest_count, following))

      traversing = following != STOP
      if traversing:
        # Choose the vaguest continuation (highest # destinations)
        # NB: only kinda sorta works
        # following = max(destinations)[1]

        # Choose the most specific continuation (lowest # destinations)
        # NB: Currently broken
        following = min(destinations)[1]

        words.append(following)
        preceding = preceding[1:] + (following,)
    return words


class Generator(object):

  def __init__(self, corpus_path, token_size=2, multi_sent=False):
    '''
    corpus_path: Path to training corpus.
    token_size: Number of words in the model's context window.

    Takes a corpus and produces a queryable model that generates sentences.
    '''

    self.token_size = token_size
    self.multi_sent = multi_sent

    with open(corpus_path) as f:
      text = f.read()

      # Relatively naive sentence splitting on punctuation
      # sentences = self.generate_sentences(text)

      # Works better but sacrifices more of the training data
      # sentences = self.generate_sentences_by_char(text)

      # Uses a pos tagged, preprocessed corpus
      sentences = self.generate_sentences_from_preprocessed(text)

      print 'Extracted {0} valid sentences from corpus.'.format(len(sentences))
      print '-----\n'

      # Save the training sentences for 'creativity' test
      self.training_words = sentences
      self.model = Model(sentences, token_size)

    # If we want to create multi-sentence output, we also need to load the
    # corresponding generalized corpus
    if multi_sent:
      ext = corpus_path.rfind('.')
      generalized_path = corpus_path[:ext] + '_generalized' + corpus_path[ext:]
      with open(generalized_path) as f:
        text = f.read()
        sentences = self.generate_sentences_from_preprocessed(text)
        self.general_model = Model(sentences, token_size)

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

  def generate_sentences_from_preprocessed(self, text):
    sentences = text.split('\n')
    return [s.split(' ') for s in sentences]

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

  def is_match(self, words1, words2):
    '''
    words1: Sentence in list form.
    words2: Sentence in list form.

    Removes out the common buzz words and checks for same words.
    '''

    # only need to filter one list since other one won't be able to match anyway
    words1 = filter(lambda w: w != 'and' and w != 'the' and w != 'a'\
       and w != 'an' and w != 'is' and w != 'for' and w != 'of' \
       and w != 'provides' and w != 'enables' and w != 'to' and w != 'of' \
       and w != 'on' and w != 'their' and w != 'company' and w != 'that' \
       and w != 'online' and w != 'mobile' and w != 'its' and w != 'application' \
       and w != 'platform' and w != 'product' and w != 'offers' and w != 'users' \
       and w != 'web' and w != 'site' and w != 'website' and w != 'app' \
       and w != 'create' and w != 'allows' and w != 'service' and w != 'services' \
       and w != 'digital' and w != 'your' and w != 'develop' and w != 'focused' \
       , words1)

    #return any(i in words1 for i in words2)
    amount = set(words1).intersection(set(words2))
    if len(amount) > 1:
      return True
    return False

  def create_sentence_from_state(self, model, preceding=None, prevWords=None):
    '''
    model: The Markov chain model from which to generate text.
    preceding: A tuple of words representing the desired starting state. Must
               be of length < self.token_size.

    Attempts to create a test-passing sentence within a certain number of tries.
    '''

    for i in range(100):
      words = model.create_sentence(preceding)

      if self.test_sentence(words):
        words = self.remove_pos_tag(words) 
        if prevWords: 
          if self.is_match(prevWords, words):
            return ' '.join(words)
        else:
          return ' '.join(words)
    return None

  def create_sentence(self, previous=None):
    '''
    previous: The sentence preceding the next one to generate, if it exists.

    Attempts to create a test-passing sentence within a certain number of tries.
    '''
    
    # Create sentence from random starting state
    sentence = self.create_sentence_from_state(self.model)

    # If we want multiple sentences, add from generalized model
    if self.multi_sent:
      start = (START,) * (self.token_size - 1) + ('We__PRP',)
      
      followup = self.create_sentence_from_state(self.general_model, start, sentence.split(' '))
      if followup:
        sentence = sentence + ' We ' + followup

    return sentence

