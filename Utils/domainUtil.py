import json
import pickle as p
import random
import numpy as np
from scipy.stats import norm


class DomainUtil:
    @staticmethod
    def get_truth_set(question_data, lookup):
        #
        truth_set = []
        for qd in question_data:
            q_set = set()
            # go through true values
            for value in qd[2]:
                q_set = q_set.union(lookup.get(qd[0]).get(value))
            truth_set.append(q_set)
        return truth_set

    @staticmethod
    def get_vocab(all_utt):
        vocabs = []
        for utt in all_utt:
            tokens = utt.split(" ")
            vocabs.extend(tokens)
        vocabs = list(set(vocabs))
        vocabs.append("EOS")
        return vocabs

    @staticmethod
    def get_prior_dist(mode, corpus_size):
        if mode == 'uniform':
            return np.ones(corpus_size) / corpus_size
        elif mode == 'gaussian':
            prob = norm.pdf(np.linspace(-2, 2, corpus_size), scale = 1.0)
            return prob / np.sum(prob)
        elif mode == 'zipf':
            prob = 1.0 / (np.arange(1,corpus_size+1) * np.log(1.78*np.arange(1,corpus_size+1)))
            return prob / np.sum(prob)
        else:
            return None

    @staticmethod
    def remove_duplicate(raw_list):
        new_list = []
        for l in raw_list:
            if not l in new_list:
                new_list.append(l)
        return new_list

    @staticmethod
    def get_actions(path):
        raw_data = p.load(open(path))
        questions = raw_data.get("questions")
        question_sets = raw_data.get("question_sets")
        return [(q[0], q[1], qs) for q, qs in zip(questions, question_sets)]

    @staticmethod
    def get_fields(corpus):
        # get all the fields in the corpus
        # return a dictionary mapping from: field -> a list of field_value
        field_dict = {}
        field_dim = {}
        for person in corpus.values():
            for field, content in person.iteritems():
                field_values = field_dict.get(field, set())
                if type(content) is list:
                    for c in content:
                        field_values.add(c)
                else:
                    field_values.add(content)
                field_dict[field] = field_values
        # convert set to list
        for field, value in field_dict.iteritems():
            field_dict[field] = list(value)
            field_dim[field] = len(value)
        return field_dict, field_dim

    @staticmethod
    def load_model(path):
        corpus = json.load(open(path, 'r'))
        print "A domain with " + str(len(corpus)) + " people."
        return corpus

    @staticmethod
    def get_lookup(corpus, field_dict):
        # create N lookup table for N field
        # each table is a mapping: field_value -> a set of person id
        # return a dict of dict for each field
        lookup = {}
        for key, person in corpus.iteritems():
            for field, content in person.iteritems():
                field_lookup = lookup.get(field, {})
                if type(content) is not list:
                    content = [content]
                for c in content:
                    person_set = field_lookup.get(c, set())
                    person_set.add(key)
                    # save the person set
                    field_lookup[c] = person_set
                # save the field_lookup
                lookup[field] = field_lookup
        return lookup



