'''
Train a markov model on color names, then use it to generate some
similar names
Inspiration from https://github.com/jsvine/markovify
'''
from os import path
import sys
import csv
import pandas as pd
from nltk import TreebankWordTokenizer
import numpy as np
import markovify
from collections import defaultdict
import pickle
import random #Choose a pattern according to weighted distribution
from color_utils import Color, rgbDistance, three_dEuclideanDistance, closestColor
from color_utils import learnColors
from color_utils import readPaintColors
from color_utils import ColorTagger
import logging

#Globals
VERBOSE_FLAG = True
tok = TreebankWordTokenizer()
myDataDir = "./data"

def tag_colors(df, basic_colors):
    '''
    pass in dataframe of color names, add a column with pos tags
    '''
    df['tag_pattern'] = 'NA'
    colorTagger = ColorTagger(basic_colors)

    for index, row in df.iterrows():

        tokens = tok.tokenize(row['color_name_raw'])

        tagged_tokens = colorTagger.tag(tokens)

        tags = [item[1] for item in tagged_tokens]

        pattern = ",".join(tags)

        final_tags = ["/".join([token, tag]) for token, tag in zip(tokens, tags)]

        df.at[index, 'tag_pattern'] = pattern

    return df

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    #Read in raw data
    logger.info("Reading in raw color names data...\n")
    color_names = readPaintColors(myDataDir)

    #Learn the 43 basic color terms and their RGB values from the data
    logger.info("Learning basic color names and RGB values...\n")
    basicColorTerms = learnColors(color_names)
    #pickle basicColorTerms
    with open('.\\data\\basicColorTerms.pyc', 'wb') as f:
        pickle.dump(basicColorTerms, f)

    #Add part of speech tags, including color name
    color_names_tagged = tag_colors(color_names, [color.name for color in  basicColorTerms])

    logger.debug("Training data matrix: {} columns, {} rows.\n".format(color_names_tagged.shape[1],
        color_names_tagged.shape[0]))


    #Convert to string buffer separated by newlines
    logger.info("Learning markov model of color names...\n")

    namesBuf = "\n".join(color_names_tagged['color_name_raw'].tolist())

    #Train markov model
    names_model = markovify.NewlineText(namesBuf, state_size = 2)
    #Pickle names_model
    #pickle basicColorTerms
    with open('.\\data\\names_model.pyc', 'wb') as f:
        pickle.dump(names_model, f)


    #Tally distribution of patterns
    logger.info("Learning counts of tag patterns of color names...\n")
    patternCounts = defaultdict(int)
    for pattern in color_names_tagged['tag_pattern']:
        patternCounts[pattern] += 1

    #pickle patternCounts
    with open('.\\data\\patternCounts.pyc', 'wb') as f:
        pickle.dump(patternCounts, f)
