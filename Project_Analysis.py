#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:33:34 2020

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


class Project_Analysis:
    def __init__(self):
        self.host_name = "localhost"
        self.port = 10000
        self.database = "default"
    
    def __getConnection(self):
        conn = hive.Connection(host = self.host_name, port = self.port, database = self.database, auth = 'NOSASL')
        return conn;
        
    def prophet_predictor(self):    
        conn = self.__getConnection()
        df = pd.read_sql('select * from crime_prophet', conn)
        conn.close()
        df.columns = ['Date', 'Block', 'Primary Type']
        df.drop(df.index[0], inplace = True)
        #Assembling a datetime by rearranging the dataframe column "Date".
        df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y %I:%M:%S %p')
        
        #Set the index to date
        df.index = pd.DatetimeIndex(df.Date)
        
        # Primary type count
        df['Primary Type'].value_counts()
        
        # Columns = ['Date', 'Crime Count']
        chicago_prophet = df.resample('M').size().reset_index()
        chicago_prophet.columns = ['Date', 'Crime Count']
        chicago_prophet_df = pd.DataFrame(chicago_prophet)
        chicago_prophet
        
        # Rename the columns
        chicago_prophet_df_final = chicago_prophet_df.rename(columns={'Date':'ds', 'Crime Count':'y'})
        chicago_prophet_df_final
        
        # Prediction
        
        m = Prophet()
        m.fit(chicago_prophet_df_final)
        
        # Forcasting into the future
        future = m.make_future_dataframe(periods=1825)
        forecast = m.predict(future)
        
        #Future prediction up to 2022
        figure = m.plot(forecast, xlabel = 'Date', ylabel = 'Crime Rate')
        
        
        #Plotting Trend Yearly
        figure3 = m.plot_components(forecast)
    
    def MLP_classifier(self):
        conn = self.__getConnection()
        df = pd.read_sql('select * from crime_classifier', conn)
        conn.close()
        df.drop(df.index[0], inplace = True)
        df.columns = ['IUCR', 'Description', 'Primary Type', 'FBI Code']
        df['IUCR'] = pd.factorize(df["IUCR"])[0]
        df['Description'] = pd.factorize(df["Description"])[0]
        df['FBI Code'] = pd.factorize(df["FBI Code"])[0]
        df['Primary Type'] = pd.factorize(df["Primary Type"])[0] 
        df['Primary Type'].unique()
        
        X = df.loc[:, ['IUCR', 'Description', 'FBI Code']].values
        y = df.loc[:, ['Primary Type']].values
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
        
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        # Neural Network
         
        nn_model = MLPClassifier(solver = 'adam', alpha = 1e-5, hidden_layer_sizes = (40,) , random_state = 1, max_iter = 2)
        nn_model.fit(X_train, y_train)
        
        y_pred = nn_model.predict(X_test) 
        
        #Predicting Accuracy and F1 Score
        from sklearn.metrics import accuracy_score, f1_score
        print("Accuracy score : ", accuracy_score(y_pred, y_test))
        print("F1_Score : ", f1_score(y_pred, y_test, average='micro'))
        
    def randomForest_classifier(self):
        conn = self.__getConnection()
        df = pd.read_sql('select * from crime_classifier', conn)
        conn.close()
        df.drop(df.index[0], inplace = True)
        
        df.columns = ['IUCR', 'Description', 'Primary Type', 'FBI Code']
        df['IUCR'] = pd.factorize(df["IUCR"])[0]
        df['Description'] = pd.factorize(df["Description"])[0]
        df['FBI Code'] = pd.factorize(df["FBI Code"])[0]
        df['Primary Type'] = pd.factorize(df["Primary Type"])[0] 
        df['Primary Type'].unique()
        
        X = df.loc[:, ['IUCR', 'Description', 'FBI Code']].values
        y = df.loc[:, ['Primary Type']].values
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
        
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        from sklearn.metrics import accuracy_score, f1_score
        print("Accuracy score : ",accuracy_score(y_pred, y_test))
        print("F1_Score : ", f1_score(y_pred, y_test, average='micro'))
    
        
    def naive_bayes_classifer(self):
        conn = self.__getConnection()
        df = pd.read_sql('select * from crime_naive_classifier', conn)
        conn.close()
        df.columns = ['Location Description', 'Ward', 'Domestic']
        df.drop(df.index[0], inplace = True)
        
        df['Location Description'] = pd.factorize(df['Location Description'])[0]
        df['Ward'] = pd.factorize(df['Ward'])[0]
        df['Domestic'] = pd.factorize(df['Domestic'])[0]
        
        df.dropna()
        
#        from sklearn.preprocessing import OneHotEncoder
#        hot = OneHotEncoder()
#        X = hot.fit_transform(df[['Location Description',  'Ward']]).toarray()
#        y = df.iloc[:, 2].values
        X = df.iloc[:, 0:2].values
        y = df.iloc[:, 2].values
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        from sklearn.naive_bayes import BernoulliNB
        classifier = BernoulliNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        accu = accuracy_score(y_test, y_pred)
        print(cm)
        print(accu)
        
    def category_plotting(self):
        conn = self.__getConnection()
        df = pd.read_sql('select * from crime_prophet', conn)
        conn.close()
        df.columns = ['Date', 'Block', 'Primary Type']
        df.drop(df.index[0], inplace = True)
        #Assembling a datetime by rearranging the dataframe column "Date".
        df.Date = pd.to_datetime(df.Date, format = '%m/%d/%Y %I:%M:%S %p')
        
        #Set the index to date
        df.index = pd.DatetimeIndex(df.Date)
        
        crimes_count_date = df.pivot_table('Block', aggfunc=np.size, columns='Primary Type', index=df.index.date, fill_value=0)
        
        plo = crimes_count_date.rolling(365).sum().plot(figsize=(16, 40), subplots=True, layout=(-1, 2), sharex=False, sharey=False)
        plt.show()
        plt.savefig('Categorical_Plotting.png')


    def crime_mapping(self):
        conn = self._getConnection()

        df = pd.read_sql("select * from crime_map", conn)
        
        df.columns = ['Location Description', 'Latitude', 'Longitude', 'Arrest', 'Community Area', 'Location']
        
        df = df.dropna()
        df.drop(df.index[0], inplace = True)
        
        #chicago_map = folium.Map(location = [41.864073,-87.706819],
        #                        zoom_start = 11,
        #                        tiles = "CartoDB dark_matter")
        
        locations = df.groupby('Community Area').first()
        
        new_locations = locations.loc[:, ['Latitude', 'Longitude', 'Location Description', 'Arrest']]
        
        
        unique_locations = df['Location'].value_counts()
        
        CR_index = pd.DataFrame({"Raw_String" : unique_locations.index, "ValueCount":unique_locations})
        CR_index.index = range(len(unique_locations))
        
        def Location_extractor(Raw_Str):
            preProcess = Raw_Str[1:-1].split(',')
            lat =  float(preProcess[0].strip('('))
            long = float(preProcess[1].strip(')'))
            return (lat, long)
        
        CR_index.drop(CR_index.index[0], inplace = True)
        CR_index['LocationCoord'] = CR_index['Raw_String'].apply(Location_extractor)
        
        CR_index  = CR_index.drop(columns=['Raw_String'], axis = 1)
        
        chicago_map_crime = folium.Map(location=[41.895140898, -87.624255632],
                                zoom_start=13,
                                tiles="CartoDB dark_matter")
        
        for i in range(500):
            lat = CR_index['LocationCoord'].iloc[i][0]
            long = CR_index['LocationCoord'].iloc[i][1]
            radius = CR_index['ValueCount'].iloc[i] / 45
            
            if CR_index['ValueCount'].iloc[i] > 1000:
                color = "#FF4500"
            else:
                color = "#008080"
            
            popup_text = """Latitude : {}<br>
                        Longitude : {}<br>
                        Criminal Incidents : {}<br>"""
            popup_text = popup_text.format(lat,
                                       long,
                                       CR_index['ValueCount'].iloc[i]
                                       )
            folium.CircleMarker(location = [lat, long], popup= popup_text,radius = radius, color = color, fill = True).add_to(chicago_map_crime)
            
            
        chicago_map_crime.save('bestmap1.html')


if __name__ == '__main__':
    p = Project_Analysis()
#    p.prophet_predictor()
#    p.MLP_classifier()
#    p.randomForest_classifier()
#    p.naive_bayes_classifer()
    p.category_plotting()
#    p.crime_mapping()