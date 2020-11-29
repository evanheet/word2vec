from input_data import InputData
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import csv
import numpy as np
import scipy.stats as ss
from nltk.metrics import spearman


class Word2Vec:
    def __init__(self, wikidump_filename, output_text_filename, emb_dimension, batch_size, window_size, iteration, initial_lr, min_count):
        
        self.data = InputData(wikidump_filename, min_count, output_text_filename)
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)
       

    def train(self):
        pair_count = self.data.evaluate_pair_count(self.window_size)
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))
        for i in process_bar:
            pos_pairs = self.data.get_batch_pairs(self.batch_size, self.window_size)
            neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 5)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]

            pos_u = Variable(torch.LongTensor(pos_u))
            pos_v = Variable(torch.LongTensor(pos_v))
            neg_v = Variable(torch.LongTensor(neg_v))
            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" % (loss.data, self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
    def calculate_probability(self, word1, word2):
        embeddings = self.skip_gram_model.u_embeddings.weight.data.numpy()
        embedding_1 = embeddings[self.data.word2id[word1]]
        embedding_2 = embeddings[self.data.word2id[word2]]
    
        numerator = np.exp(np.sum(embedding_1*embedding_2))
        denominator = np.sum(np.exp(np.sum(np.multiply(embedding_1, embeddings), axis=1)), axis=0)
    
        return(numerator/denominator)
    
    def wordsim353_spearman(self, input_filename):
        target_word = []
        context_word = []
        human_scores = []
        with open(input_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            ws353_pairs = -1
            for row in csv_reader:
                if ws353_pairs == -1:
                    ws353_pairs += 1
                else:
                    target_word.append(row[0])
                    context_word.append(row[1])
                    human_scores.append(float(row[2]))
                    ws353_pairs += 1
                
        for pair in range(0, ws353_pairs):
            if (target_word[pair] not in self.data.word2id):
                raise Exception('Target word not in model vocab: ', target_word[pair])
            if (context_word[pair] not in self.data.word2id):
                raise Exception('Context word not in model vocab: ', context_word[pair])
                
        human_rankings = ss.rankdata(human_scores)
        
        machine_scores = []
        for pair in range(0, len(human_scores)):
            machine_scores.append(self.calculate_probability(target_word[pair], context_word[pair]))
        machine_rankings = ss.rankdata(machine_scores)

        human_scores_dict = dict()
        machine_scores_dict = dict()
        for pair in range(0, len(human_scores)):
            human_scores_dict[pair] = human_rankings[pair]
            machine_scores_dict[pair] = machine_rankings[pair]
            
        return spearman.spearman_correlation(human_scores_dict, machine_scores_dict)
            
            