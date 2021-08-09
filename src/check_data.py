import pandas as pd
import utils

def check_data():
    
    columns = utils.get_columns()

    df = pd.read_csv("data/covid_data_1.csv", names=columns)
    print(df.head()) 
    print(df.info())

if __name__ == "__main__":
    check_data()