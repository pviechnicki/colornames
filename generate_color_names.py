import sys
import random
import pickle
from color_utils import Color
from color_utils import closestColor
from color_utils import closestThreeColors
from nltk import TreebankWordTokenizer
from color_utils import ColorTagger
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

def satisfiesConstraint(candidate, refPattern, refColorList, colorTagger):
    '''
    Test a given color name candidate against the pattern and desired basic color term,
    returning True if it satisfies
    '''
    result = False
    tokens = tok.tokenize(candidate)
    tagged_tokens = colorTagger.tag(tokens)
    candidatePattern = [item[1] for item in tagged_tokens]

    #If COLOR not in pattern then just test whether two lists of POS tags are the same.
    result = listCompare(candidatePattern, refPattern)

    #Else test that plus color term match
    if 'COLOR' in refPattern and result == True:
        colorTermPosition = [i for i,pos in enumerate(refPattern) if pos == 'COLOR'][0]

        if tokens[colorTermPosition] not in refColorList:
            result = False
    return result

if __name__ == '__main__':

    #Read saved training data on counts of names with each POS pattern
    sys.stderr.write("Reading POS pattern counts...\n")
    with open(patternCountsFilename, 'rb') as f:
        allPatternCounts = pickle.load(f)
        patternCounts = {pattern: count for (pattern, count) in allPatternCounts.items() if 'COLOR' in pattern.split(",")}

    #Read basic color terms dictionary
    sys.stderr.write("Reading basic color terms...\n")
    with open(basicColorTermsFilename, 'rb') as f:
        basicColorTerms = pickle.load(f)

    #Read markov names_model
    sys.stderr.write("Reading markov language model...\n")
    with open(markovFilename, 'rb') as f:
        names_model = pickle.load(f)

    #Choose pattern according to distro
    total = sum(patternCounts.values())
    patternString = random.choices([c for c in patternCounts.keys()], [count/total for count in patternCounts.values()])[0]
    pattern = patternString.split(",")

    #Create color tagger
    basic_colors = [color.name for color in  basicColorTerms]
    colorTagger = ColorTagger(basic_colors)

    while True:
        root = tk.Tk()
        style = ttk.Style(root)
        style.theme_use('clam')

        rgb = askcolor((300, 100, 100), root)
        testColor = Color(rgb[1], rgb[0][0], rgb[0][1], rgb[0][2])

        basic_color_family = [color[1].name for color in closestThreeColors(testColor, basicColorTerms)]

        #Generate some names
        sys.stderr.write("Generating candidate names using pattern {} for color ({}, {}, {})...\n".format(pattern, testColor.r, testColor.g, testColor.b))
        satisfied = False
        iterations = 0
        while satisfied != True and iterations <= MAX_ITERATIONS:
            candidate = names_model.make_sentence(test_output=False) #Adjust parameter for overlap ratio
            iterations += 1
            sys.stderr.write("\tTesting candidate {}...\n".format(candidate))
            satisfied = satisfiesConstraint(candidate, pattern, basic_color_family, colorTagger)


        if satisfied:
            print("Found a good name! {}".format(candidate))
        else:
            print("Sorry - couldn't find a good candidate in {} iterations.".format(MAX_ITERATIONS))
