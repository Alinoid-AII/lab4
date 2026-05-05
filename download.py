import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    df = pd.read_csv('heart.csv')
    return df

def clear_data(df):
    cat_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    num_columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease']
    
    # Анализ и очистка данных
    
    # здравый смысл
    question_engine = df[(df["RestingBP"] < 30) | (df["RestingBP"] > 250)] #уровень артериального давления у человека в сознании не м.б < 50  и > 250
    df = df.drop(question_engine.index)
    
    # здравый смысл
    question_engine = df[(df["Cholesterol"] == 0) | (df["Cholesterol"] > 600)] # то же для холестирина
    df = df.drop(question_engine.index) 
    
    # здравый смысл
    question_price = df[(df["Age"] <= 0) | (df["Age"] > 120)]
    df = df.drop(question_price.index)

    # здравый смысл
    question_price = df[~df['Sex'].isin(['M', 'F'])]
    df = df.drop(question_price.index)    
    
    df = df.reset_index(drop=True)  
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns]);
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    df.to_csv('df_clear.csv', index=False)
    return True

if __name__ == "__main__":
    data = download_data()
    clear_data(data)
    print("Data processing completed successfully!")
