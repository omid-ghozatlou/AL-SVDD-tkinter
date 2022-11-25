# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:22:30 2021

@author: CEOSpaceTech
"""


import os
import os.path
import pandas as pd
import numpy as np
import glob
def makescv(path):
    # path = 'D:/Omid/UPB/SVM/Galaxy-classification-master/data/california/'
    Train_p = path+'/Train/positive/'
    Train_n = path+'/Train/negative/'
    unlabeled = path+'/unlabeled/'
    Test_p = path+'/Test/positive/'
    Test_n = path+'/Test/negative/'
    
    # images=[i for i in files_in_train if i in files_in_annotated]
    
    class_train_p = pd.DataFrame(glob.glob(os.path.join(Train_p,'*')))
    class_train_n = pd.DataFrame(glob.glob(os.path.join(Train_n,'*')))
    class_u = pd.DataFrame(glob.glob(os.path.join(unlabeled,'*')))
    class_test_p = pd.DataFrame(glob.glob(os.path.join(Test_p,'*')))
    class_test_n = pd.DataFrame(glob.glob(os.path.join(Test_n,'*')))
    
    label0 = pd.DataFrame(np.zeros([len(class_train_p),1]))
    label1 = pd.DataFrame(np.zeros([len(class_train_n),1]))+1
    label2 = pd.DataFrame(np.zeros([len(class_u),1]))+3
    label3 = pd.DataFrame(np.zeros([len(class_test_p),1]))
    label4 = pd.DataFrame(np.zeros([len(class_test_n),1]))+1
    
    # test set
    test_set_p=pd.concat([class_test_p, label3], axis=1)
    test_set_n=pd.concat([class_test_n, label4], axis=1)
    test_set=pd.concat([test_set_p, test_set_n], axis=0)
    test_set.to_csv(path+'/test.csv', header=["Image", "Labels"], index=False)
    # Train set
    class0=pd.concat([class_train_p, label0], axis=1)
    class1=pd.concat([class_train_n, label1], axis=1)
    train_set=pd.concat([class0, class1], axis=0)
    train_set.to_csv(path+'/train.csv', header=["Image", "Labels"], index=False)
    # Apply model set
    apply_set=pd.concat([class_u, label2], axis=1)
    apply_set.to_csv(path+'/apply_model.csv',header=["Image", "Labels"], index=False)
    return len(class_u)
    
