#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:32:19 2020

@author: student
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:23:17 2020

@author: student
"""
import requests
from hdfs import InsecureClient
from pyhive import hive
import sasl
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy as np


class Project_ETL:
    def __init__(self):
        self.host_name = "localhost"
        self.port = 10000
        self.database = "default"
    
    def __getConnection(self):
        conn = hive.Connection(host = self.host_name, port = self.port, database = self.database, auth = 'NOSASL')
        return conn;
    
    def download_data():
        url = 'https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD'
        r = requests.get(url)
        
        with open('/home/student/Project/Dataset.csv', 'wb') as f:
            f.write(r.content)

    def load_data_to_hdfs():
        try:
            file = '/home/student/Project/Dataset.csv'
            
        except:
            print("File not found")
        
        hdfsclient = InsecureClient("http://localhost:50070", user = "hduser")
        hdfs_path = "/"
        hdfsclient.upload(hdfs_path, file)	#Dumping file into the hadoop
        
        hdfs_path = "/project"
        hdfsclient.upload(hdfs_path, file)
            
    def load_data_to_hive(self):
        conn = self.__getConnection()
        cur = conn.cursor()
        
        query_create = """Create table crime(ID int, Case_Number string, Crime_Date string, Block string, IUCR string, Primary_Type string,
                                             Description string, Location_Description string, Arrest string, Domestic string, Beat string,
                                             District string, Ward int, Community_Area int, FBI_Code string, X_Coordinate int,
                                             Y_Coordinate float, Year int, Updated_On string, Latitude float, Longitude float,
                                             Location string) row format delimited fields terminated by "," stored as textfile"""
        cur.execute(query_create)
        
        query_create = """Create table crime1(ID int, Case_Number string, Crime_Date string, Block string, IUCR string, Primary_Type string, Description string, Location_Description string, Arrest string, Domestic string, Beat string, District string, Ward int, Community_Area int, FBI_Code string, X_Coordinate int, Y_Coordinate float, Year int, Updated_On string, Latitude float, Longitude float, Location string) ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' stored as textfile tblproperties ('skip.header.line.count'='1')"""
        cur.execute(query_create)
        
        query_load = "load data inpath '/Dataset.csv' OVERWRITE INTO TABLE crime"
        cur.execute(query_load)
        
        query_load = "load data inpath '/project/Dataset.csv' OVERWRITE INTO TABLE crime1"
        cur.execute(query_load)
        
        query_create = 'create table crime_classifier as (select IUCR, Description, Primary_Type, FBI_Code from crime)'
        cur.execute(query_create)
        
        query_create = 'create table crime_prophet as (select Crime_Date, Block, Primary_Type from crime)'
        cur.execute(query_create)
        
        query_create = 'create table crime_naive_classifier as (select Location_Description, Ward, Domestic from crime)'
        cur.execute(query_create)
        
        query_create = 'create table crime_district as (select district, primary_type from crime where district is not null and id is not null and district!="false")'
        cur.execute(query_create)
        
        query_create = 'create table crime_map as (select Location_Description, latitude, longitude, arrest, community_area from crime where latitude is not null and longitude is not null and arrest is not null and community_area is not null and location_description is not null)'
        cur.execute(query_create)    
    
        cur.close()
        conn.close()

if __name__ == 'main':
    p = Project_ETL()
    p.download_data()
    p.load_data_to_hdfs()
    p.load_data_to_hive()
    