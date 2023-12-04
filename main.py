import nltk
from nltk.corpus import brown
import collections

# Load the Brown Corpus and split it into sentences
nltk.download('brown')
nltk.download('universal_tagset')
sentences = brown.tagged_sents(tagset='universal')

# Split data into training and testing sets
train_data = sentences[:int(0.8 * len(sentences))]
test_data = sentences[int(0.8 * len(sentences)):]

# Create a list of all unique words and tags in the corpus
words = [word for sent in train_data for word, _ in sent]
tags = [tag for sent in train_data for _, tag in sent]
unique_words = list(set(words))
unique_tags = list(set(tags))

# Calculate emission probabilities
word_tag_count = collections.defaultdict(lambda: collections.defaultdict(int))
tag_count = collections.defaultdict(int)

for sentence in train_data:
    for word, tag in sentence:
        word_tag_count[word][tag] += 1
        tag_count[tag] += 1

word_given_tag = {}
for word in unique_words:
    word_given_tag[word] = {tag: (word_tag_count[word][tag] + 1e-10) / tag_count[tag] for tag in unique_tags}

# Calculate transition probabilities
transition_probabilities = collections.defaultdict(lambda: collections.defaultdict(int))

for sentence in train_data:
    for i in range(len(sentence) - 1):
        prev_tag = sentence[i][1]
        next_tag = sentence[i + 1][1]
        transition_probabilities[prev_tag][next_tag] += 1

tag_transition_prob = {}
for tag in unique_tags:
    tag_transition_prob[tag] = {next_tag: (transition_probabilities[tag][next_tag] + 1e-10) / tag_count[tag]
                                for next_tag in unique_tags}

# Calculate initial state distribution
initial_state_distribution = collections.defaultdict(int)
for sentence in train_data:
    initial_state_distribution[sentence[0][1]] += 1

total_initial = sum(initial_state_distribution.values())
initial_state_distribution = {tag: (count + 1e-10) / total_initial for tag, count in initial_state_distribution.items()}

# Viterbi Algorithm with HMM
def viterbi(words, unique_words, unique_tags, word_given_tag, tag_transition_prob, initial_state_distribution):
    V = [{}]
    backpointer = [{}]

    # Initialization step with initial state distribution
    for tag in unique_tags:
        V[0][tag] = initial_state_distribution.get(tag, 1e-10) * word_given_tag.get(words[0], {}).get(tag, 1e-10)
        backpointer[0][tag] = None

    # Recursion step
    for t in range(1, len(words)):
        V.append({})
        backpointer.append({})
        for tag in unique_tags:
            probabilities = {
                prev_tag: V[t - 1][prev_tag] * word_given_tag.get(words[t], {}).get(tag, 1e-10) *
                           tag_transition_prob[prev_tag].get(tag, 1e-10)
                for prev_tag in unique_tags
            }
            max_prob_tag = max(probabilities, key=probabilities.get)
            V[t][tag] = probabilities[max_prob_tag]
            backpointer[t][tag] = max_prob_tag

    # Termination step
    best_seq_end = max(V[-1], key=V[-1].get)
    best_seq = [best_seq_end]

    # Backtrace to find the best path
    for t in range(len(words) - 1, 0, -1):
        best_seq_end = backpointer[t][best_seq_end]
        best_seq.insert(0, best_seq_end)

    return best_seq

# Testing the Viterbi algorithm on the test set
correct = 0
total = 0

for sentence in test_data:
    words = [word for word, _ in sentence]
    tags = [tag for _, tag in sentence]

    predicted_tags = viterbi(words, unique_words, unique_tags, word_given_tag, tag_transition_prob, initial_state_distribution)

    for i in range(len(tags)):
        if tags[i] == predicted_tags[i]:
            correct += 1
        total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
