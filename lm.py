# imports go here
import random
import sys
import math
import numpy as np

from statistics import mean
from collections import Counter

"""
Don't forget to put your name and a file comment here
Karan Satwani
HW-2
CS-6120
"""

class LanguageModel:
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing):
        """Initializes an untrained LanguageModel
    Parameters:
      n_gram (int): the n-gram order of the language model to create
      is_laplace_smoothing (bool): whether or not to use Laplace smoothing
    """
        # Initializing parameters
        self.n_gram = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing

        # Initializing variables
        self.n_gram_list = []
        self.vocab_count = Counter()
        self.n_minus_vocab_count = Counter()
        self.uni_vocab = Counter()

        # initializing count for perplexity
        self.perp_sentence_count = 0

    def make_ngrams(self, tokens: list, n: int) -> list:
        """Creates n-grams for the given token sequence.
    Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create
    Returns:
    list: nested list of strings, each list being one of the individual n-grams
    """
        n_gram_list = []
        for i in range(len(tokens) - n + 1):
            temp_word = []
            for j in range(n):
                temp_word.append(tokens[i + j])
            n_gram_list.append(' '.join(temp_word))
        return n_gram_list

    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Parameters:
      training_file_path (str): the location of the training data to read

    Returns:
    None
    """
        file = open(training_file_path, "r")
        contents = file.read()
        file.close()

        # splitting contents and putting in list and storing in dictionary with a count
        splitted = contents.split()
        self.vocab_count = Counter(splitted)

        # replacing every word seen one time with <UNK>
        for i in range(len(splitted)):
            if self.vocab_count[splitted[i]] == 1:
                splitted[i] = self.UNK
        self.uni_vocab = splitted

        # making n_grams and counting the vocabulary
        if self.n_gram >= 2:  # i.e. more than bi-gram, we would need n-1 gram for probabilities
            self.n_minus_vocab_count = Counter(self.make_ngrams(splitted, (self.n_gram - 1)))

        # making n grams on the list
        self.n_gram_list = self.make_ngrams(splitted, self.n_gram)
        self.vocab_count = Counter(self.n_gram_list)
        pass

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
    Parameters:
      sentence (str): a sentence with tokens separated by whitespace to calculate the score of
    Returns:
      float: the probability value of the given string for this model
    """
        prob = 1
        tokens = sentence.split()

        # replacing every word/token seen in the sentence one time with <UNK>
        for i in range(len(tokens)):
            if tokens[i] not in self.uni_vocab:
                tokens[i] = self.UNK

        # creating n grams of the tokens
        sentence_gram_list = self.make_ngrams(tokens, self.n_gram)
        self.perp_sentence_count = len(sentence_gram_list)

        for gram in sentence_gram_list:
            # doing a reverse split because we only need the n-1 gram for the given gram in loop
            # that is the vocab and is used for probability
            n_minus_gram = gram.rsplit(' ', 1)[0]

            # if gram is not the vocabulaty, then its count i.e. numerator here is 0
            if (gram not in self.vocab_count):
                numerator = 0
            else:
                numerator = self.vocab_count.get(gram)
            if self.is_laplace_smoothing:
                if self.n_gram == 1:
                    prob = prob * ((numerator + 1) / (len(self.n_gram_list) + len(self.vocab_count)))
                else:
                    # taking n-1 gram to check if it si present in the vocab to calculate the proba

                    # the vocabulary for n_gram > 1 would consist of tokens of size n-1 grams
                    prob = prob * (numerator + 1) / \
                           (self.n_minus_vocab_count.get(n_minus_gram) + len(self.n_minus_vocab_count))

            # no laplace smoothing
            else:
                if self.n_gram == 1:
                    prob = prob * numerator / (len(self.n_gram_list))
                else:
                    # the vocabulary for n_gram > 1 would consist of tokens of size n-1 grams
                    prob = prob * numerator / \
                           self.n_minus_vocab_count.get(n_minus_gram)
        return prob

    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.
        Returns:
        str: the generated sentence
        :rtype: object
        """
        # picking the gram starting with <s>
        start_list = []
        for i in range(len(self.n_gram_list)):
            if self.n_gram_list[i][0:3] == "<s>":
                start_list.append(self.n_gram_list[i])

        # picking a random start
        random_gram = random.choice(start_list)
        sentence = random_gram

        if self.n_gram == 1:
            # keep adding tokens until we reach end
            while sentence[-4:] != "</s>":
                random_gram = random.choice(self.n_gram_list)
                # not adding starting sentence character again
                if random_gram != "<s>":
                    sentence += " " + random_gram
        else:
            while True:
                # creating an empty list, will be populated with possible next grams
                # will pick random from those
                temp_list = []
                # we would only later half of the gram to pick next one
                # because we only pick one uni-gram every step and add it to the sentence
                random_gram = random_gram.split(' ', 1)[1]
                for i in range(len(self.n_gram_list)):
                    # will only append the gram but using part of the previous as starting point
                    if self.n_gram_list[i][0:len(random_gram)] == random_gram:
                        temp_list.append(self.n_gram_list[i])
                random_gram = random.choice(temp_list)
                adding_token = random_gram.split(' ')
                # only adding a new token to the already existing sentence basically adding a uni-gram
                sentence += " " + adding_token[self.n_gram - 1]
                # cheking if the sentence has ended, if yes, we break from the loop
                if sentence[-4:] == "</s>":
                    break

        # adding extra <s> and </s> to the sentence
        if self.n_gram > 2:
            intro = (self.n_gram - 2) * '<s> '
            outro = (self.n_gram - 2) * ' </s>'
            return intro + sentence + outro

        else:
            return sentence

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
    Parameters:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing strings, one per generated sentence
    """
        list = []
        for i in range(n):
            list.append(self.generate_sentence())
        return list

    def perplexity(self, test_sequence):
        """Measures the perplexity for the given test sequence with this trained model.
                 As described in the text, you may assume that this sequence
                 may consist of many sentences "glued together".
        Parameters:
          test_sequence (string): a sequence of space-separated tokens to measure the perplexity of
        Returns:
          float: the perplexity of the given sequence
        """

        # probab =
        # for i in range(10):
        #     print(test_sequence[i])
        #     print(self.vocab_count)
        #     probab = probab * self.score(test_sequence[i])

        # calling score function and calculating perplexity using log
        perplex = math.log10(self.score(test_sequence))
        new_perplex = pow(10, perplex)
        new_perplex = pow(new_perplex, (-1 / (self.perp_sentence_count)))

        return new_perplex


def main():
    # TODO: implement
    training_path = sys.argv[1]
    testing_path1 = sys.argv[2]
    testing_path2 = sys.argv[3]

    # opening and reading the two testing files
    file1 = open(testing_path1, "r", encoding='utf-8')
    contents1 = file1.read()
    file1.close()
    file2 = open(testing_path2, "r", encoding='utf-8')
    contents2 = file2.read()
    file2.close()

    test_filepath1 = contents1.split('\n')
    test_filepath2 = contents2.split('\n')

    # initiating language models
    uni_model = LanguageModel(1, True)
    bi_model = LanguageModel(2, True)
    # training both models
    uni_model.train(training_path)
    bi_model.train(training_path)

    # creating 4 list which will store probabilities of a model each
    prob_uni1 = []
    prob_uni_my = []
    prob_bi1 = []
    prob_bi_my = []
    for i in range(len(test_filepath1)):
        prob_uni1.append(uni_model.score(test_filepath1[i]))
        prob_bi1.append(bi_model.score(test_filepath1[i]))

    for i in range(len(test_filepath2)):
        prob_uni_my.append(uni_model.score(test_filepath2[i]))
        prob_bi_my.append(bi_model.score(test_filepath2[i]))

    # calculating average probabilties for all the models
    average_prob1 = mean(prob_uni1)
    average_prob_uni_my = mean(prob_uni_my)
    average_prob3 = mean(prob_bi1)
    average_prob_bi_my = mean(prob_bi_my)

    # calculating standard deviation for all the probabilities
    deviation1 = np.std(prob_uni1)
    deviation_uni_my = np.std(prob_uni_my)
    deviation3 = np.std(prob_bi1)
    deviation_bi_my = np.std(prob_bi_my)

    # creating string sentences for which perplexity will be calculated
    perplex_1 = " ".join(test_filepath1[0:10])
    perplex_2 = " ".join(test_filepath2[0:10])

    print("Model: unigram, laplace smoothed\nSentences:")
    sentences_uni = uni_model.generate(50)
    for i in range(len(sentences_uni)):
        print(sentences_uni[i])
    print("test corpus: " + testing_path1 + "\n")
    print('\n# of sentences: ' + str(len(test_filepath1)))
    print('\nAverage probability: ' + str(average_prob1))
    print('\nStandard Deviation: ' + str(deviation1))
    print("\ntest corpus: " + testing_path2)
    print('\n# of sentences: ' + str(len(test_filepath2)))
    print('\nAverage probability: ' + str(average_prob_uni_my))
    print('\nStandard Deviation: ' + str(deviation_uni_my))

    print("\nModel: bigram, laplace smoothed\nSentences:")
    sentences_bi = bi_model.generate(50)
    for i in range(len(sentences_bi)):
        print(sentences_bi[i])
    print("test corpus: " + testing_path1 + "\n")
    print('\n# of sentences: ' + str(len(test_filepath1)))
    print('\nAverage probability: ' + str(average_prob3))
    print('\nStandard Deviation: ' + str(deviation3))
    print("\ntest corpus: " + testing_path2 )
    print('\n# of sentences: ' + str(len(test_filepath2)))
    print('\nAverage probability: ' + str(average_prob_bi_my))
    print('\nStandard Deviation: ' + str(deviation_bi_my))

    print('\nPerplexity for uni_grams:')
    print('hw2-test.txt: ' + str(uni_model.perplexity(perplex_1)))
    print('hw2-my-test.txt: ' + str(uni_model.perplexity(perplex_2)))

    print('\nPerplexity for bi_grams:')
    print('hw2-test.txt: ' + str(bi_model.perplexity(perplex_1)))
    print('hw2-my-test.txt: ' + str(bi_model.perplexity(perplex_2)))
    pass


if __name__ == '__main__':

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) != 4:
        print("Usage:", "python lm.py training_file.txt testingfile1.txt testingfile2.txt")
        sys.exit(1)

    main()
