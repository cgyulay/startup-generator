
'''
Takes a corpus of preprocessed sentences and attempts to replace company
names with pronouns, allowing for generation of better follow-up sentences
in multi-sentence generation.
'''

import nltk
from pattern.en import conjugate, lemma, lexeme

class DataAugment(object):

  def __init__(self, corpus_path, dest_path):
    with open(corpus_path) as f:
      text = f.read()
      sentences = text.split('\n')
      dest = open(dest_path, 'w')
      converted = 0

      # Identify sentences that match replacement pattern, generalize these
      # sentences, and write to file
      for i, s in enumerate(sentences):
        words = s.split(' ')
        if self.pattern_match(words):
          converted += 1
          modified = self.generalize(words)
          dest.write(' '.join(modified) + '\n')

      # Finish
      rate = converted / float(len(sentences)) * 100
      print 'Successfully generalized {0:.2f}% of sentences from corpus.'.format(rate)
      print 'Saved {0} new sentences to {1}.'.format(converted, dest_path)
      dest.close()

  def pattern_match(self, words):
    if len(words) < 2: return False

    # The pattern we're looking for is "company_name singular_verb"
    pos1 = self.extract_pos_tag(words[0])
    pos2 = self.extract_pos_tag(words[1])

    return pos1 == 'NNP' and pos2 == 'VBZ'

  def generalize(self, words):
    # Convert to generalized sentence (not company specific)
    we = 'We__PRP'
    words[0] = we

    # Pluralize the verb using pattern library
    pl = conjugate(self.remove_pos_tag(words[1]), 'pl') + '__VBP'
    words[1] = pl
    return words

  def remove_pos_tag(self, word):
    i = word.find('__')
    return word[:i]

  def extract_pos_tag(self, word):
    i = word.find('__') + 2
    return word[i:]

# Run
corpus_path = 'tagged_corpora/crunchbase_descriptions_2000.txt'
ext = corpus_path.rfind('.')
dest_path = corpus_path[:ext] + '_generalized' + corpus_path[ext:]
augment = DataAugment(corpus_path, dest_path)