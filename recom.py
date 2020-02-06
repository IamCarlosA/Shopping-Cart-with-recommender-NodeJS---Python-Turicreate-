from flask import Flask, jsonify
from flask_cors import CORS 
from pymongo import MongoClient
import pandas as pd
import turicreate as tc
from flask import request

app = Flask(__name__)
CORS(app)

client = MongoClient('localhost', 27017)
db = client['shopping-cart']
collection = db['recoms']

exclude_data = {'_id': False, '__v': False}
raw_data = list(collection.find({}, projection=exclude_data))
data_df = pd.DataFrame(raw_data)


data = tc.SFrame(data_df)

model = tc.factorization_recommender.create(
    data, 'userId', 'itemId', target='rating')


def create_output(model_recommendation, users_to_recommend):    
    
    recomendation = model_recommendation.recommend(users_to_recommend)

    
    df_rec = tc.SFrame.to_dataframe(recomendation)
   
    
    df_rec = df_rec.drop(['score','rank'],axis=1) 
    
    df_rec['recommendedProducts'] = df_rec['itemId'].groupby(df_rec['userId']).transform(lambda x: ','.join(x.astype(str))).drop_duplicates()
          
    df_output = df_rec[['userId', 'recommendedProducts']].drop_duplicates().set_index('userId').dropna()                       
    
    return df_output

@app.route('/users')
def recommend_new_user():
        
    df_recomendation = create_output(model, ['-1']).iloc[0].to_dict()

    return jsonify(df_recomendation)

@app.route('/user')
def recommend_user():
        
    user=request.args.get('user')
    
    df_recomendation = create_output(model, [user]).iloc[0].to_dict()

    return jsonify(df_recomendation)

if __name__ == '__main__':
    app.debug = True
    app.run()