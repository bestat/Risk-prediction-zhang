# -*- coding: utf-8 -*-
'''
pick up the vaule of the model output for each sopt. output is npy file.
'''


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np

if __name__=='__main__':
    '''step0  load the excel by panda '''
    df = pd.read_excel('181219norimen-zhang.xlsx', sheet_name='transformed_new!')
    print("Column headings:")
    print(df.columns)
    '''step1 extract the location names and labels'''
    name = df['Key2'][:1345]
    normal_location = name.notnull()

    label = df['Key172'][:1345]
    '''step2 extract other attributes'''
    cut_bank = df['Key8'][:1345][normal_location]
    shape = df[['Key18','Key19','Key20','Key21','Key22','Key23','Key24','Key25',
                'Key26']][:1345][normal_location]
    structure = df[['Key27','Key28','Key29','Key30','Key35']][:1345][normal_location]
    Intensity_of_soil = df['Key36'][:1345][normal_location]
    material_of_bank  = df['Key37'][:1345][normal_location]
    
    name = name.where(name.notnull(), '000')
    
    #'''step2 use new dataset and sheet name is : transformed_new!'''
    
    '''step2  unique the locations, eliminate the '000'locations, and keep one label 
        for one location. The rule is :
            if there is at least one landslide occures in thate location, set label to 1 
            to that location.
    '''
    new_names=[]
    new_labels=[]
    acu_label = 0
    for i in range(len(name)-1):
        if name[i] != '000' and name[i+1] !='000':
            new_names.append(name[i])
            new_labels.append(label[i]) 
        elif name[i] != '000' and name[i+1] =='000':
            new_names.append(name[i])
            acu_label = acu_label + label[i]
        elif name[i] =='000' and name[i+1] != '000':
            acu_label = acu_label+label[i]
            new_labels.append(acu_label)
            acu_label = 0
        elif name[i] =='000' and name[i+1] == '000':
            acu_label = acu_label+label[i]
        else:
            break
    if len(new_names) > len(new_labels): # the last one
        new_labels.append(acu_label)    
    new_names = np.array(new_names)
    new_labels = np.array(new_labels)
    new_labels = 1*(new_labels>0)
    
    '''step3 find other attributes'''
    features = {}
    for i in range(len(new_names)):
        feature = [cut_bank[i]]
        feature = feature + shape.iloc[i].tolist()
        feature = feature + structure.iloc[i].tolist()
        feature = feature + [Intensity_of_soil[i], material_of_bank[i]]
        features[new_names[i]] = feature
    '''step3 find all the labels'''    
    output = {}
    for out, label in zip(new_names, new_labels):
        output[out] = label
    '''step3 save the dictionary as labels.txt'''
    f = open('labels.txt','w')
    f.write(str(output))
    f.close()
    '''ste4 save the dictioinary'''
    f = open('features.txt','w')
    f.write(str(features))
    f.close()
   