from __future__ import division
import argparse
import pandas as pd

import numpy as np
from collections import defaultdict
from scipy.special import expit

import matplotlib.pyplot as plt
import pickle
from scipy.spatial import distance

__authors__ = ['Asrorbek ORZIKULOV', 'Venkata Shivaditya MEDURI',
               'Raghuwansh RAJ', 'Sauraj VERMA']
__emails__ = ['asrorbek.orzikulov@essec.edu', 'meduri.venkatashivaditya@essec.edu',
              'raghuwansh.raj@essec.edu', 'sauraj.saurajverma@essec.edu']


def text2sentences(path):
    sentencesList = []
    with open(path, "r") as file:
        for line in file:
            sentence = line.lower().split()
            validElements = []
            for item in sentence:
                if item.isalpha() or item == "'t":  # Remove strings containing punctuations and numbers
                    validElements.append(item)
            sentencesList.append(validElements)
    return sentencesList


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


def createVocabulary(sentences, minCount):
    corpus = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            corpus[word] += 1

    wordFreqDict = {word: count for word, count in corpus.items()
                    if count >= minCount}  # Removing words with small frequency
    w2id = {word: idx for (idx, word) in enumerate(wordFreqDict)}
    w2id['UNKN'] = len(w2id)
    return w2id, wordFreqDict


def noiseDistribution(word2Frequency, word2Id):
    newFrequencies = {}
    sumNewFreq = 0
    for word in word2Frequency:
        wordIdx = word2Id[word]
        wordFreq = word2Frequency[word]
        newFreq = wordFreq ** 0.75
        newFrequencies[wordIdx] = newFreq
        sumNewFreq += newFreq

    keys_list = []
    probs_list = []
    for idx, freq in newFrequencies.items():
        keys_list.append(idx)
        probs_list.append(freq / sumNewFreq)
    return keys_list, probs_list


def sigmoid(x):
    return expit(np.clip(x, -100, 100))


class SkipGram:
    def __init__(self, sentences=list(), dimEmbedding=100, negativeRate=5, winSize=5, minCount=1, lr=0.01, epochs=2):
        self.w2id, self.vocab = createVocabulary(sentences, minCount)
        self.trainSet = sentences
        self.winSize = winSize
        self.keys_list, self.values_list = noiseDistribution(self.vocab, self.w2id)
        self.epochs = epochs
        self.dimEmbedding = dimEmbedding
        self.lr = lr
        self.inputWeights = np.random.rand(len(self.w2id), self.dimEmbedding)
        self.contextWeights = np.random.rand(len(self.w2id), self.dimEmbedding)
        self.negativeWords = negativeRate
        self.trainWords = 0
        self.accLoss = 0

    def sample(self, contextWords):
        # samples negative words, omitting those in set omit
        # The words are sampled from unigram model raised to the power of 3/4
        negatives = []
        for i in range(self.negativeWords):
            negative = np.random.choice(self.keys_list, p=self.values_list)
            while negative in contextWords:
                negative = np.random.choice(self.keys_list, p=self.values_list)
            negatives.append(negative)
        return negatives

    def loss(self, word, posWord, negWords):
        lossValue = -np.log(sigmoid(np.dot(self.inputWeights[word], self.contextWeights[posWord])))
        for negWord in negWords:
            lossValue -= np.log(sigmoid(-np.dot(self.inputWeights[word], self.contextWeights[negWord])))
        return lossValue

    def train(self):        
        losses = []
        for epoch in range(self.epochs):
            print(f" > Epoch {epoch + 1} started!")
            for counter, sentence in enumerate(self.trainSet, start=1):
                for wordPos, word in enumerate(sentence):
                    if word not in self.w2id:
                        word = 'UNKN'
                    wIdx = self.w2id[word]
                    start = max(0, wordPos - self.winSize)
                    end = min(wordPos + self.winSize + 1, len(sentence))
                    contextWords = set(sentence[start:end])
                    for context_word in contextWords:
                        if context_word not in self.w2id:
                            context_word = 'UNKN'
                        ctxtId = self.w2id[context_word]
                        if ctxtId == wIdx:
                            continue
                        negativeIds = self.sample(contextWords)
                        loss = self.loss(wIdx, ctxtId, negativeIds)
                        self.trainWord(wIdx, ctxtId, negativeIds)
                        self.trainWords += 1
                        self.accLoss += loss

                if counter % 100 == 0:
                    print(f" > line {counter} of {len(self.trainSet)} done!")
                    losses.append(self.accLoss / self.trainWords)
                    self.trainWords = 0
                    self.accLoss = 0
        plt.plot(losses)
        plt.savefig("losses.png")
        return losses

    def trainWord(self, wordId, posId, negIds):
        h = self.inputWeights[wordId]
        con_p = self.contextWeights[posId]  # Positive context word vector
        grad_inp_w = (sigmoid(np.dot(con_p, h)) - 1) * con_p
        grad_con_p = (sigmoid(np.dot(con_p, h)) - 1) * h
        for negId in negIds:
            con_n = self.contextWeights[negId]
            self.contextWeights[negId] -= self.lr * sigmoid(np.dot(con_n, h)) * h
            grad_inp_w += sigmoid(np.dot(con_n, h)) * con_n
        self.inputWeights[wordId] -= self.lr * grad_inp_w
        self.contextWeights[posId] -= self.lr * grad_con_p

    def save(self, path):
        f = open(path, "wb")
        model = {"w2id": self.w2id, "weights": self.inputWeights}
        pickle.dump(model, f)
        f.close()
        return

    def similarity(self,word1,word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        if word1 not in self.w2id.keys():
            word1 = "UNKN"
        if word2 not in self.w2id.keys():
            word2 = "UNKN"
        w1 = self.inputWeights[self.w2id[word1]]
        w2 = self.inputWeights[self.w2id[word2]]
        similarity = 1 - distance.cosine(w1, w2)
        return similarity

    def load(self, path):
        a_file = open(path, "rb")
        model = pickle.load(a_file)
        self.w2id = model["w2id"]
        self.inputWeights = model["weights"]
        a_file.close()
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')
    opts = parser.parse_args()
    if not opts.test:
        sentences_list = text2sentences(opts.text)
        sg = SkipGram(sentences_list)
        sg.train()
        sg.save(opts.model)
    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram()
        sg.load(opts.model)
        for a, b, _ in pairs:
            print(sg.similarity(a, b))
