#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:49:13 2020

@author: student
"""
from Project import Project

def main():
    p = Project()
    
#    p.download_data()
#    p.load_data_to_hdfs()
#    p.load_data_to_hive()
    p.MLP_classifier()
#    p.prophet_predictor()
#    p.randomForest_classifier()
#    p.naive_bayes_classifer()
    
main()


