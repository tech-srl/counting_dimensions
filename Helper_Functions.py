import itertools
import random

def mean(num_list):
    return sum(num_list)*1.0/len(num_list)

def n_words_of_length(n,length,alphabet):
    if 50*n >= pow(len(alphabet),length):
        res = all_words_of_length(length, alphabet)
        random.shuffle(res)
        return res[:n]  
    #else if 50*n < total words to be found, i.e. looking for 1/50th of the words or less
    res = set()
    while len(res)<n:
        word = ""
        for _ in range(length):
            word += random.choice(alphabet)
        res.add(word)
    return list(res)

def all_words_of_length(length,alphabet):
    return [''.join(list(b)) for b in itertools.product(alphabet, repeat=length)]

def compare_network_to_classifier(rnn,classifier,words):
    n = len(words)
    count = 0
    for w in words:
        if rnn.classify_word(w) == classifier(w):
            count += 1
    return count/n

def map_nested_dict(d,mapper):
    if not isinstance(d,dict):
        return mapper(d)
    return {k:map_nested_dict(d[k],mapper) for k in d}

def nice_number_str(val,digits=2):
    res = str(int(val*pow(10,digits))/pow(10,digits))
    d = len(res.split(".")[1])
    if d<digits:
        res += ("0"*(digits-d))
    return res

class MissingInput(Exception):
    pass