import markov

print 'Building model...this could take a few seconds.\n'

# NB: Generator's __init__ in markov.py must be modified per the directions
# there if you select an untagged corpus.
# corpus_path = 'corpora/crunchbase_descriptions.txt'
# corpus_path = 'corpora/crunchbase_descriptions_small.txt'
# corpus_path = 'corpora/punctuation_test.txt'
corpus_path = 'tagged_corpora/crunchbase_descriptions_50000.txt'

token_size = 3
multi_sent = False
num_ideas = 5

model = markov.Generator(corpus_path, token_size, multi_sent=multi_sent)
for i in range(num_ideas):
  s = model.create_sentence()
  if s != None: print s + '\n'