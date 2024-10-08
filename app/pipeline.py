import dill
import pandas as pd
import numpy as np
import requests
import urllib.parse
import json
import datetime
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def new_features(df_sessions):
    import pandas as pd
    import requests
    df_features = df_sessions.copy()

    # органический трафик
    organic_tr = ['organic', 'referal', '(none)']
    df_features.loc[df_features.utm_medium.isin(organic_tr), 'organic_traffic'] = 1
    df_features.organic_traffic = df_features.organic_traffic.fillna(0)

    # реклама в соц сетях
    social_media = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                    'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    df_features.loc[df_features.utm_source.isin(social_media), 'social_media_adv'] = 1
    df_features.social_media_adv = df_features.social_media_adv.fillna(0)

    # Россия ли страна
    df_features.loc[df_features.geo_country == 'Russia', 'is_Russia'] = 1
    df_features.is_Russia = df_features.is_Russia.fillna(0)

    # Месяц
    df_features['visit_date'] = pd.to_datetime(df_features['visit_date'])
    df_features['visit_month'] = df_features.visit_date.dt.month

    # Калндарный день
    df_features['visit_day'] = df_features.visit_date.dt.day

    # День недели
    df_features['visit_weekday'] = df_features['visit_date'].apply(lambda x: x.isoweekday())

    # Час визита
    df_features['visit_hour'] = pd.to_datetime(df_features['visit_time'], format="%H:%M:%S").dt.hour

    # Высота и ширина экрана (2)
    df_features.device_screen_resolution = df_features.device_screen_resolution.replace(['(not set)'],
                                                                                        df_features.device_screen_resolution.mode())
    df_features['device_screen_length'] = df_features.device_screen_resolution.apply(lambda x: int(x.split('x')[0]))
    df_features['device_screen_width'] = df_features.device_screen_resolution.apply(lambda x: int(x.split('x')[1]))

    # долгота и широта
    cities_list = df_features.geo_city.unique().tolist()
    cities_dict = dict.fromkeys(cities_list)

    headers = {'User-Agent': 'my user agent'}

    for k, v in cities_dict.items():
        try:
            url = "https://nominatim.openstreetmap.org/?addressdetails=1&q=" + k + "&format=json&limit=1"
            response = requests.get(url, headers=headers, timeout=None).json()
            cities_dict[k] = [response[0]["lat"], response[0]["lon"]]
        except:
            country = df_features[df_features.geo_city == k].geo_country.mode()[0]
            url = "https://nominatim.openstreetmap.org/?addressdetails=1&q=" + country + "&format=json&limit=1"
            response = requests.get(url, headers=headers, timeout=None).json()
            cities_dict[k] = [response[0]["lat"], response[0]["lon"]]
    df_features.loc[:, 'lat'] = df_features.geo_city.apply(
        lambda x: cities_dict[x][0] if x in cities_dict.keys() else NaN)
    df_features.loc[:, 'lon'] = df_features.geo_city.apply(
        lambda x: cities_dict[x][1] if x in cities_dict.keys() else NaN)
    df_features.lon = df_features.lon.astype(float)
    df_features.lat = df_features.lat.astype(float)

    return df_features


def drop_col(df_sessions):
    df_drop = df_sessions.copy()

    df_drop.drop(['session_id', 'client_id', 'visit_date', 'visit_time', 'device_model',
                  'device_browser', 'geo_city'], axis=1, inplace=True)
    return df_drop


def encoding(df_sessions):
    from sklearn.preprocessing import OrdinalEncoder
    import pandas as pd
    df_enc = df_sessions.copy()

    categorical_columns = df_enc.select_dtypes(['object']).columns.tolist()
    encoder = OrdinalEncoder()
    df_enc[categorical_columns] = encoder.fit_transform(df_enc[categorical_columns])
    return df_enc


def impute_func(df_sessions):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LinearRegression
    import pandas as pd

    df = df_sessions.copy()

    iter_imputer = IterativeImputer(initial_strategy='median', estimator=LinearRegression(),
                                    max_iter=20, min_value=0)
    df_imp = iter_imputer.fit_transform(df)
    df_imp = pd.DataFrame(df_imp, columns=df.columns)
    return df_imp


def main():
    df_hits = pd.read_csv('data/ga_hits-001.csv', low_memory=False)
    df_sessions = pd.read_csv('data/ga_sessions.csv', low_memory=False)

    # add y
    list_of_ca = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                  'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                  'sub_submit_success', 'sub_car_request_submit_click']

    df_hits.loc[(df_hits.event_action.isin(list_of_ca) == True), 'cr'] = 1
    df_hits.loc[(df_hits.event_action.isin(list_of_ca) == False), 'cr'] = 0

    df_cr_groupped = df_hits.groupby(['session_id']).agg({'cr': max})
    df_sessions = pd.merge(left=df_sessions, right=df_cr_groupped, on='session_id', how='inner')

    X = df_sessions.drop('cr', axis=1)
    y = df_sessions['cr']

    preprocessor = Pipeline(steps=[
        ('new_features', FunctionTransformer(new_features)),
        ('filter', FunctionTransformer(drop_col)),
        ('encode', FunctionTransformer(encoding)),
        ('imputer', FunctionTransformer(impute_func))
    ])

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgboost.XGBClassifier(n_estimators=150, max_depth=10,
                                             learning_rate=0.06439300974204062, subsample=0.7118205006700702,
                                             colsample_bytree=0.7570898823208844,
                                             colsample_bylevel=0.6001157653038874,
                                             colsample_bynode=0.5673619876100888, objective='binary:logistic',
                                             eval_metric='auc'))
    ])
    pipe.fit(X, y)

    score = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
    print(f'model: {type(pipe.named_steps["classifier"]).__name__}, roc_auc_score: {score:.4f}')

    with open('Pickled_model.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'project',
                'author': 'Bogomaz Ekaterina',
                'type': type(pipe.named_steps["classifier"]).__name__,
                'roc_auc_score': score
            }
        }, file)


if __name__ == '__main__':
    main()