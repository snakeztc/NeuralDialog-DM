import json
import pickle as p
import random


class DomainUtil:
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



