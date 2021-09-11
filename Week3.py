'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random
import math


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):

    # TODO
    
    targetValues = df[df.columns[-1]].values
    valueCounter = dict()
    
    for value in targetValues:
        if value not in ValueCounter.keys():
            valueCounter[value] = 1
        else
            valueCounter[value] += 1
    
    targets = list(valueCounter.values())
    entropy = 0
    
    for value in targets:
        prop = value/(sum(targets))
        entropy += (-1)*prop*math.log(prop,2)
    
    # DONE
    
    return entropy
    

def get_entropy_of_value(dic):
    targets = list(dic.values())
    entropy = 0
	
    for value in targets:
        prop = value/(sum(targets))
        entropy += (-1)*prop*math.log(prop,2)
    
    return entropy
		


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):

    # TODO
    
    targetValues = list[df[df.columns[-1]].values]
    attrValues = list[df[attribute].values]
    
    valueCounter = dict()
    total = len(attrValues)
    
    for value in range(len(attrValues)):
        if attrValues[value] not in valueCounter.keys():
            valueCounter[attrValues[value]] = dict()
            if targetValues[value] not in valuecounter[attrValues[value]].keys():
                valueCounter[attrValues[value]][targetValues[value]] = 1
            else:
                valueCounter[attrValues[value]][targetValues[value]] += 1
        else:
            if targetValues[value] not in valuecounter[attrValues[value]].keys():
                valueCounter[attrValues[value]][targetValues[value]] = 1
            else:
                valueCounter[attrValues[value]][targetValues[value]] += 1
    
    avg_info = 0
    for value in valueCounter:
        avg_info += (sum(valueCounter.values()/total) * get_entropy_of_value(valueCounter[value])
    
    # DONE
    
    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    
    # TODO
    
    information_gain = get_entropy_of_dataset(df) - get_avg_info_of_attribute(df, attribute)
        
    # DONE
    
    return information_gain


#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    
    # TODO
    
    df_ig = dict()
    selected = df.columns[0]
    
    for col in df.columns[:-1]:
        if col not in df_ig.keys():
            df_ig[col] = get_information_gain(df, col)
            if df_if(col) > df_ig(selected):
                selected = col
    
    return(df_ig, selected)
    
    # DONE
    
    pass
