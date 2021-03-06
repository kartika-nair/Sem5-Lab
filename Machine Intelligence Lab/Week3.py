'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random
# import math


def entropy_dataset_helper(arr, entropy):   
    for i in arr:
        p = i/(sum(arr))
        entropy += (-1) * p * np.log2(p)
    return entropy
    

'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):

    # TODO
    
    df.dropna(axis = 0)
    
    cols = list(df.columns)
    str = cols[-1]
    values_under_attribute = df[str].tolist()
    
    entropy = 0
    counterDictionary = dict()
    
    for i in values_under_attribute:
        if i not in counterDictionary.keys():
            counterDictionary[i] = 1
        else:
            counterDictionary[i] += 1
    
    arr = list(counterDictionary.values())
    entropy = entropy_dataset_helper(arr, entropy)
    
    # DONE
    
    # print(entropy)
    
    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):

    # TODO
    values = df[attribute].unique()
    counter = len(df.index)

    entropy = 0
    for i in values:
        individual = df.loc[df[attribute] == i]
        attr_counter = len(individual.index)
        entropy_dataset = get_entropy_of_dataset(individual)
        entropy +=  (attr_counter/counter) * entropy_dataset
    
    avg_info = abs(entropy)
        
    # print(avg_info)
    
    return avg_info
    
    # DONE


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    
    # TODO
    
    information_gain = abs(get_entropy_of_dataset(df) - get_avg_info_of_attribute(df, attribute))
        
    # DONE
    
    # print(information_gain)
    
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
    
    dictionary_final = dict()
    col = df.columns[0]
    
    for i in df.columns[:-1]:
        if i not in dictionary_final.keys():
            dictionary_final[i] = get_information_gain(df, i)
        if dictionary_final[i] > dictionary_final[col]:
            col = i
    
    # print(dictionary_final, col)
    
    return(dictionary_final, col)
    
    # DONE


'''
outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(',')
temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(',')
humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(',')
windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE'.split(',')
play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')
dataset = {'outlook': outlook, 'temp': temp, 'humidity': humidity, 'windy': windy, 'play': play}
df = pd.DataFrame(dataset, columns=['outlook', 'temp', 'humidity', 'windy', 'play'])
# print(get_entropy_of_dataset(df))
# print(get_avg_info_of_attribute(df, 'temp'))
# print(get_information_gain(df, 'temp'))
# print(get_selected_attribute(df))
'''
