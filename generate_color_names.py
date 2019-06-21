import sys
import random
import pickle
import logging
from color_utils import Color
from color_utils import closestColor, closestNColors
from nltk import TreebankWordTokenizer
from color_utils import ColorTagger, findRelatedWords
import tkinter as tk
import tkinter.ttk as ttk
from tkcolorpicker import askcolor

tok = TreebankWordTokenizer()
MAX_ITERATIONS = 500

#Names of pickled training data
patternCountsFilename = '.\\data\\patternCounts.pyc'
basicColorTermsFilename = '.\\data\\basicColorTerms.pyc'
markovFilename = '.\\data\\names_model.pyc'

def listCompare(listA, listB):
    '''
    Return true if two lists contain exactly the same elements in the same order
    '''
    return listA == listB

def satisfiesConstraint(candidate, refPattern, refColors, colorTagger):
    '''
    Test a given color name candidate against the pattern and desired basic color terms,
    returning True if it satisfies
    Satisfies constraint if POSPattern matches and if semantics matches
    '''
    matchesPOSPattern = False
    matchesSemantics = False

    tokens = tok.tokenize(candidate)

    tagged_tokens = colorTagger.tag(tokens)

    candidatePattern = [item[1] for item in tagged_tokens]

    #First test whether part of speech pattern in candidate matches reference pattern
    matchesPOSPattern = listCompare(candidatePattern, refPattern)

    #Then, test whether the semantics match depending on whether it contains a basic color term or not
    if matchesPOSPattern == True:
        if 'COLOR' in refPattern:

            colorTermPosition = [i for i,pos in enumerate(refPattern) if pos == 'COLOR'][0]

            #Semantic match for a color term is exact match against the three closest basic color terms
            if tokens[colorTermPosition] in [color.name for color in refColors]:
                matchesSemantics = True
        else:

            #Semantic match for a pattern with no color term means looking for a related term in the pattern
            #Test each related term to see whether it occurs in the candidate
            for relatedTerm in refColor.getRelatedWords():
                if relatedTerm in tokens:
                    matchesSemantics = True
                    break

    return (matchesPOSPattern==True and matchesSemantics==True)

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    #Read saved training data on counts of names with each POS pattern
    logging.info("Reading POS pattern counts...\n")
    with open(patternCountsFilename, 'rb') as f:
        allPatternCounts = pickle.load(f)
        patternCounts = {pattern: count for (pattern, count) in allPatternCounts.items() if 'COLOR' in pattern.split(",")}


    #Read basic color terms dictionary
    logging.info("Reading basic color terms...\n")
    with open(basicColorTermsFilename, 'rb') as f:
        basicColorTerms = pickle.load(f)

    #Read markov names_model
    logging.info("Reading markov language model...\n")
    with open(markovFilename, 'rb') as f:
        names_model = pickle.load(f)

    #Create color tagger
    basic_colors = [color.name for color in  basicColorTerms]
    colorTagger = ColorTagger(basic_colors)

    while True:
        root = tk.Tk()
        style = ttk.Style(root)
        style.theme_use('clam')

        rgb = askcolor((255, 255, 0), root)
        testColor = Color(rgb[1], rgb[0][0], rgb[0][1], rgb[0][2])

        closeColors = closestNColors(testColor, basicColorTerms, 1)

        logging.debug(":".join([c.name for c in closeColors]) + "\n")

        #Choose pattern according to distro
        total = sum(patternCounts.values())
        patternString = random.choices([c for c in patternCounts.keys()], [count/total for count in patternCounts.values()])[0]
        pattern = patternString.split(",")

        #Generate some names
        logging.info("Generating candidate names using pattern {} for color ({}, {}, {})...\n".format(pattern, testColor.r, testColor.g, testColor.b))
        satisfied = False
        iterations = 0
        while satisfied != True and iterations <= MAX_ITERATIONS:
            candidate = names_model.make_sentence(test_output=False) #Adjust parameter for overlap ratio
            iterations += 1
            logging.info("\tTesting candidate {}...\n".format(candidate))
            satisfied = satisfiesConstraint(candidate, pattern, closeColors, colorTagger)


        if satisfied:
            print("Found a good name for input color R:{}, G:{}, B:{} -- {}".format(testColor.r, testColor.g, testColor.b, candidate))
        else:
            print("Sorry - couldn't find a good candidate for input color R:{}, G:{}, B:{}in {} iterations.".format(testColor.r, testColor.g, testColor.b, MAX_ITERATIONS))
