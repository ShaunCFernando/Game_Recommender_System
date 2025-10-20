import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv('video_games.csv')
data = data.drop(['NA_players', 'EU_players', 'JP_players', 'Other_players', 'Global_players', 'Critic_Count', 'User_Score', 'User_Count', 'Developer'], axis=1)

data['Rating'].fillna('na', inplace=True)
data['Critic_Score'].fillna('na', inplace=True)

data['Year_of_Release'] = data['Year_of_Release'].astype(str)
data['Name'] = data['Name'].str.lower()
data['Publisher'] = data['Publisher'].str.lower()
data['Platform'] = data['Platform'].str.lower()
data['Genre'] = data['Genre'].str.lower()
data['Rating'] = data['Rating'].str.lower()

data['Name'] = data['Name'] + ' (' + data['Platform'] + ')'

# Combine relevant features into a single string
features_combined = data[data.columns[2:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)), 
    axis=1
)

vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(features_combined)

similarities = cosine_similarity(vectorized)
data_vect = pd.DataFrame(similarities, columns=data['Name'], index=data['Name']).reset_index()
data_vect = data_vect.drop(['Name'], axis=1)
data_vect = pd.concat([data, data_vect], axis=1)

while True:
    name = input("Enter the name of a game you like in the format 'Game (Platform)': ")
    name = name.lower()
    print('Wait a minute and you will see our recommendations. Enjoy! ')
    try:
        recommended_data = data_vect[data_vect[name] > 0.6]
        recommended = pd.concat([recommended_data['Name'], recommended_data['Year_of_Release'], recommended_data['Publisher'], recommended_data['Genre'], recommended_data['Critic_Score'], recommended_data['Rating']], axis=1)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(recommended)
    except:
        print('Sorry, we can not find a suitable match. Try a different game! ')
    retry = input("Do you want to try another game? (yes/no): ")
    if(retry != 'yes'):
        print("Goodbye!")
        break