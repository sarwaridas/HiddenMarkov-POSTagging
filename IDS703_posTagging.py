import nltk
import numpy as np

"""
Use the first 10k tagged sentences from the Brown corpus to generate the
components of a part-of-speech hidden markov model: the transition
matrix, observation matrix, and initial state distribution. Use the universal
tagset:
nltk.corpus.brown.tagged_sents(tagset=’universal’)[:10000]
Also hang on to the mappings between states/observations and indices. Include an OOV observation and smoothing everywhere
"""


def genTags(corpus):
    """
    Returns unique set of words and POS tags
    """
    wordTags = set()
    posTags = set()
    for line in corpus:
        for (word, tag) in line:
            wordTags.add(word)
            posTags.add(tag)
    wordTags.add("OOV")
    return sorted(wordTags), sorted(posTags)


def gen_initial_state_dist(corpus):
    """
    Given a corpus, finds the inital state for each tagging
    Return a dicitionary of smoothed probability for each tagging
    """
    wordMap, tagMap = genTags(corpus)
    initial_prob = np.zeros(len(tagMap))
    for i in corpus:
        initial_prob[tagMap.index(i[0][1])] += 1
        pass
    initial_prob += 1
    initial_prob = initial_prob / np.sum(initial_prob, axis=0, keepdims=True)
    return initial_prob


def gen_transition_matrix(corpus):
    """
    Given a corpus, returns a transition matrix using the POS tagging
    """
    wordMap, tagMap = genTags(corpus)
    trans = np.zeros((len(tagMap), len(tagMap)))
    for i in corpus:
        for row, col in zip(i[:-1], i[1:]):
            trans[tagMap.index(row[1]), tagMap.index(col[1])] += 1
    trans += 1
    trans = trans / np.sum(trans, axis=1, keepdims=True)
    return trans


def gen_observation_matrix(corpus):
    """
    Given a corpus, finds the observation matrix
    """
    wordMap, tagMap = genTags(corpus)
    obs = np.zeros((len(tagMap), len(wordMap)))
    for i in corpus:
        for word in i:
            obs[
                tagMap.index(word[1]),
                wordMap.index(word[0]),
            ] += 1
    obs += 1
    obs = obs / np.sum(obs, axis=1, keepdims=True)
    return obs


"""

Implement a function viterbi() that takes arguments:
1. obs - the observations [list of ints]
2. pi - the initial state probabilities [list of floats]
3. A - the state transition probability matrix [2D numpy array]
4. B - the observation probability matrix [2D numpy array]
and returns:
states - the inferred state sequence [list of ints]

"""


def viterbi(obs, pi, A, B):
    """
    obs = list of index
    pi = initial state probability
    A = transition matrix
    B = observation matrix
    """
    A = np.log(A)  # make sure it's log scale to avoid underflow
    B = np.log(B)  # make sure it's log scale to avoid underflow
    initial_prob = np.log(pi)  # make sure it's log scale to avoid underflow
    I = A.shape[0]  # Number of states
    N = len(obs)  # Length of observation sequence
    # viterbi = cumulative probability matrix
    # backtrace = backtrace matrix
    viterbi = np.zeros((I, N))
    backtrace = np.zeros((I, N - 1))
    # initial for the first word
    viterbi[:, 0] = initial_prob + B[:, obs[0]]
    for n in range(1, N):
        for i in range(I):
            temp_product = A[:, i] + viterbi[:, n - 1] + B[i, obs[n]]
            viterbi[i, n] = np.max(temp_product)
            backtrace[i, n - 1] = np.argmax(temp_product)
    # Backtracking
    S_opt = np.zeros(N)
    # start from the last word with largest cumulative probability
    S_opt[-1] = np.argmax(viterbi[:, -1])
    for n in range(N - 2, -1, -1):
        S_opt[n] = backtrace[int(S_opt[n + 1]), n]
    S_opt = [int(x) for x in S_opt]
    return S_opt


"""
Infer the sequence of states for sentences 10150-10152 of the Brown corpus 
nltk.corpus.brown.tagged_sents(tagset=’universal’)[10150:10153]
and compare against the truth.
"""


def infer_sequence(
    test, transition_matrix, observation_matrix, initial_prob, wordMap, tagMap
):
    my_ouput = list(range(len(test)))
    correct_output = list(range(len(test)))
    for i, sentence in enumerate(test):
        obs = [
            wordMap.index(word[0]) if word[0] in wordMap else (len(wordMap) - 1)
            for word in sentence
        ]
        my_ouput[i] = viterbi(
            obs, initial_prob, transition_matrix, observation_matrix
        )  # getting states from vertibi function
        my_ouput[i] = [tagMap[j] for j in my_ouput[i]]  # tagging states
        correct_output[i] = [word[1] for word in sentence]
    return my_ouput, correct_output


if __name__ == "__main__":
    corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
    wordMap, tagMap = genTags(corpus)
    mat_state = gen_transition_matrix(corpus)
    mat_obs = gen_observation_matrix(corpus)
    init_state = gen_initial_state_dist(corpus)
    corpus_test = nltk.corpus.brown.tagged_sents(tagset="universal")[
        10150:10153
    ]  # test data
    predictions, correct = infer_sequence(
        corpus_test, mat_state, mat_obs, init_state, wordMap, tagMap
    )
