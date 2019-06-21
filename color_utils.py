'''
Helper functions and classes for generating color names
'''

#Imports
import sys
from math import sqrt
import pandas as pd
from os import path
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer
from nltk import FreqDist
from nltk import TaggerI
from nltk import pos_tag
from nltk.corpus import wordnet as wn


#Globals
VERBOSE_FLAG = True
tok = TreebankWordTokenizer()


class Color():
    '''
    Holds color name and rgb values, and related words
    '''
    def __init__(self, name, r, g, b):
        self.name = name
        self.r = r
        self.g = g
        self.b = b
        self.related_words = [] #List of related words with similar meanings

    def setRelatedWords(self, newRelatedWords):
        '''
        Set the related words for the color object
        '''
        assert isinstance(newRelatedWords, list)

        self.related_words = newRelatedWords

    def getRelatedWords(self):
        '''
        Return the related_words list
        '''
        return (self.related_words)

def rgbDistance(color1, color2):
    '''
    pass it two Color objects for color 1 and 2, and it gives you back the distance in 3-d space
    '''
    return three_dEuclideanDistance((color1.r, color1.g, color1.b), (color2.r, color2.g, color2.b))

def three_dEuclideanDistance(tuple1, tuple2):
    '''
    3-d Euclidean distance
    '''
    distance = sqrt((tuple1[0] - tuple2[0])**2 + (tuple1[1] - tuple2[2])**2 + (tuple1[2] - tuple2[2])**2)
    return distance

def closestColor(newColor, referenceColors):
    '''
    pick the color out of the reference color list which is the closest match to the new color
    '''
    minDistance = rgbDistance(newColor, referenceColors[0])
    result = referenceColors[0]
    for referenceColor in referenceColors[1:]:
        distance = rgbDistance(newColor, referenceColor)
        if distance < minDistance:
            minDistance = distance
            result = referenceColor
    return result

def closestNColors(newColor, referenceColors, n):
    '''
    Pick the top n closest colors to the new color from the reference color list
    '''
    distances = [rgbDistance(newColor, referenceColor) for referenceColor in referenceColors]
    colorsWithDistances = zip(distances, referenceColors)
    result = sorted(colorsWithDistances, key=lambda tup: tup[0])[:n]
    return ([c[1] for c in result])

def learnColors(colorNamesDf, n=43):
    '''
    Learn the basic color names and their RGB values from the data frameself.
    Return list of (basic_color_term objects: (name + rgb tuple))
    '''
    words = []
    results = []

    red_n = defaultdict(int)
    red_sum = defaultdict(int)
    green_n = defaultdict(int)
    green_sum = defaultdict(int)
    blue_n = defaultdict(int)
    blue_sum = defaultdict(int)

    for index, row in colorNamesDf.iterrows():
        tokens = tok.tokenize(row['color_name_raw'])

        #Store data for average rgb values per token
        for t in tokens:
            red_n[t] += 1
            red_sum[t] += row['red']

            green_n[t] += 1
            green_sum[t] += row['green']

            blue_n[t] += 1
            blue_sum[t] += row['blue']

        words += tokens

    fd = FreqDist(words)

    basic_color_terms = [c for (c, f) in fd.most_common(43)]
    drop = ['of', 'mist', 'sea', 'sweet', 'spring', 'ice', 'sky', 'light', 'garden', 'stone', 'deep',
    'golden', 'dark', 'pale', 'soft', 'the', 'fresh', 'mountain', 'sage', 'desert']
    basic_color_terms = [color for color in basic_color_terms if color not in drop]

    #Take the average R, G, B values for each of the basic color terms
    for basic_color_term in basic_color_terms:
        r = red_sum[basic_color_term]/red_n[basic_color_term]
        g = green_sum[basic_color_term]/green_n[basic_color_term]
        b = blue_sum[basic_color_term]/blue_n[basic_color_term]
        newColor = Color(basic_color_term, r, g, b)

        #Add related words to the basic color object
        relatedWords = findRelatedWords(basic_color_term)

        newColor.setRelatedWords(relatedWords)

        results.append(newColor)

    return results

def readPaintColors(dataDir):
    '''
    Four csv files, one per company, read and pass back as dataframe
    pass in data directory root
    '''

    paint_companies = ['sherwinwilliams', 'behr', 'resene', 'benjaminmoore']

    dataframes = []

    for paint_company in paint_companies:
        filename = path.normpath(dataDir + '\\'+ paint_company + '\\db.csv')
        temp = pd.read_csv(filename, names= ['color_name_raw', 'red', 'green', 'blue'])

        dataframes.append(temp)

        if VERBOSE_FLAG:
            sys.stderr.write("Read {} lines from {}\n".format(len(temp), paint_company))

    color_names = pd.concat(dataframes)

    #Add a period at the end of each line
    color_names['color_name_clean'] = color_names['color_name_raw'].astype(str) + '.'

    #Add a rowid
    color_uid = 0
    color_names['uid'] = 0
    for index, row in color_names.iterrows():
        color_names.at[index, 'uid'] = color_uid
        color_uid += 1

    return color_names

class ColorTagger(TaggerI):
    '''
    Pos Tagger that replaces color names with a special tag 'COLOR'
    '''
    def __init__(self, color_names):
        self.color_names = color_names

    def tag(self, tokens):
        '''
        Tag tokens with part of speech or 'color' if token is on reference color list
        '''
        tagged_tokens = pos_tag(tokens)
        color_tagged_tokens = []
        for (token, tag) in tagged_tokens:
            if token in self.color_names:
                new_tag = 'COLOR'
            else:
                new_tag = tag
            color_tagged_tokens.append((token, new_tag))
        return color_tagged_tokens

def get_synset_tokens(wnSynset):
    '''
    Pass in a wordnet synset object
    and it gives you back a set of all the unique tokens in the words of the synset
    '''
    #Split off the first field, using period as delimiter
    name = wnSynset.name().split(".")[0]

    tokens = set(name.split("_"))

    return tokens


def get_hyponyms(wnSynset):
    '''
    Pass in a wordnet synset, and it finds all the hyponyms. Recursive function.
    Uses set union (|=)
    source: https://stackoverflow.com/questions/15330725/how-to-get-all-the-hyponyms-of-a-word-synset-in-python-nltk-and-wordnet?rq=1
    '''

    hyponyms = set()

    for hyponym in wnSynset.hyponyms():

        hyponyms |= set(get_hyponyms(hyponym))

    return hyponyms | set(wnSynset.hyponyms())

def findRelatedWords(word):
    '''
    Use nltk.wordnet.synsets to expand a word to find related terms
    Pass in a string, get back a list of related terms
    '''

    assert(isinstance(word, str))

    words = set()
    hyponyms = []

    temp = [get_hyponyms(s) for s in wn.synsets(word)]
    #Filter out empty sets from synsets with no hyponyms
    temp2 = [i for i in temp if len(i) > 0]
    #Collapse/flatten into a single set
    hyponyms = set()
    for i in temp2:
        hyponyms |= i

    for h in hyponyms:
        words |= get_synset_tokens(h)

    return(list(words))
