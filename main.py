import markov

print 'Building model...this could take a few seconds.\n'

# corpus_path = 'corpora/crunchbase_descriptions.txt'
corpus_path = 'corpora/crunchbase_descriptions_small.txt'
# corpus_path = 'tagged_corpora/crunchbase_descriptions_2000.txt'
# corpus_path = 'corpora/punctuation_test.txt'
token_size = 3

model = markov.Generator(corpus_path, token_size)
for i in range(5):
  s = model.create_sentence()
  if s != None: print s + '\n'