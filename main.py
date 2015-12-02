import markov

print 'Building model...this could take a few seconds.\n'

corpus_path = 'corpora/crunchbase_descriptions.txt'
# corpus_path = 'corpora/crunchbase_descriptions_small.txt'
# corpus_path = 'corpora/punctuation_test.txt'
token_size = 2

model = markov.Generator(corpus_path, token_size)
for i in range(5):
  print model.create_sentence() + '\n'