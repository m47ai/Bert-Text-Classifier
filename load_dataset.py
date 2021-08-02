import pandas as pd 

def load_data(file_name):
    

    data_raw = pd.read_csv(file_name,sep = ",")

    data = data_raw.iloc[:,[1,3]]

    data.columns = ['text','tag']

    data['tag'].fillna('Random_Tag',inplace = True)

    data['one_hot'] = [list((row[1].values))for  row in pd.get_dummies(data['tag']).iterrows()]

    return data[['text','one_hot']]



 









