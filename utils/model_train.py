import pandas as pd



def read_csv():
    data = pd.read_csv('result.csv')
    return data

def pearson_correlation():
    data = read_csv()
    corr = data.corr()
    return corr

if __name__ == '__main__':
    data = read_csv()
    #print(data['pre_status'])
    print(pearson_correlation())