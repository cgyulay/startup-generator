import markovify

# Get raw text as string.
with open('corpora/crunchbase_descriptions.txt') as f:
  text = f.read()

# Build the model
print 'Building model...this could take a few seconds.\n'

state_size = 2
text_model = markovify.Text(text, state_size=state_size)

# Print five randomly-generated sentences
for i in range(10):
  print(text_model.make_sentence())
  print '\n'

# Print three randomly-generated sentences of no more than 300 characters
# for i in range(3):
  # print(text_model.make_short_sentence(300))