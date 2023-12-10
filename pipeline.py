import dill
import pandas as pd
import requests
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


def new_features(df_sessions):
    import pandas as pd
    import requests
    df_features = df_sessions.copy()

    def checking(df, list_of_values):
        if df in list_of_values:
            return True
        else:
            return False

    # Органический трафик
    organic_tr = ['organic', 'referal', '(none)']
    df_features.loc[:, 'organic_traffic'] = df_features.utm_medium.apply(
        lambda x: 1 if checking(x, organic_tr) == True else 0)
    # Реклама в соц сетях
    social_media = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                    'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    df_features.loc[:, 'social_media_adv'] = df_features.utm_source.apply(
        lambda x: 1 if checking(x, social_media) == True else 0)
    # Россия ли страна
    df_features.loc[:, 'is_Russia'] = df_features.geo_country.apply(lambda x: 1 if x == 'Russia' else 0)
    # Фича месяц
    df_features['visit_month'] = pd.to_datetime(df_features['visit_date']).dt.month
    # Фича день
    df_features['visit_day'] = pd.to_datetime(df_features['visit_date']).dt.day
    # Фича день недели
    df_features['visit_weekday'] = pd.to_datetime(df_features['visit_date']).apply(lambda x: x.isoweekday())
    # Фича час визита
    df_features['visit_hour'] = pd.to_datetime(df_features['visit_time'], format="%H:%M:%S").dt.hour
    # Фичи высоты и ширины экрана (2)
    df_features.device_screen_resolution = df_features.device_screen_resolution.replace(['(not set)'],
                                                                                        df_features.device_screen_resolution.mode())
    df_features['device_screen_length'] = df_features.device_screen_resolution.apply(lambda x: int(x.split('x')[0]))
    df_features['device_screen_width'] = df_features.device_screen_resolution.apply(lambda x: int(x.split('x')[1]))
    # долгота и широта
    cities_list = df_features.geo_city.unique().tolist()
    cities_dict = dict.fromkeys(cities_list)
    for k, v in cities_dict.items():
        try:
            url = "https://nominatim.openstreetmap.org/?addressdetails=1&q=" + k + "&format=json&limit=1"
            response = requests.get(url).json()
            cities_dict[k] = [response[0]["lat"], response[0]["lon"]]
        except:
            country = df_features[df_features.geo_city == k].geo_country.mode()[0]
            url = "https://nominatim.openstreetmap.org/?addressdetails=1&q=" + country + "&format=json&limit=1"
            response = requests.get(url).json()
            cities_dict[k] = [response[0]["lat"], response[0]["lon"]]
    df_features.loc[:, 'lat'] = df_features.geo_city.apply(
        lambda x: cities_dict[x][0] if x in cities_dict.keys() else 0)
    df_features.loc[:, 'lon'] = df_features.geo_city.apply(
        lambda x: cities_dict[x][1] if x in cities_dict.keys() else 0)
    df_features.lon = df_features.lon.astype(float)
    df_features.lat = df_features.lat.astype(float)

    return df_features


def drop_columns(df_sessions):
    df_drop = df_sessions.copy()
    columns_to_drop = ['session_id', 'client_id', 'visit_date', 'visit_time', 'utm_source', 'utm_keyword', 'utm_medium',
                       'utm_campaign', 'utm_adcontent', 'device_category', 'device_model', 'device_brand',
                       'device_browser', 'device_os', 'geo_country', 'geo_city', 'device_screen_resolution']
    return df_drop.drop(columns_to_drop, axis=1)


def main():
    df_hits = pd.read_csv('data/ga_hits-001.csv')
    df_sessions = pd.read_csv('data/ga_sessions.csv')

    list_of_ca = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                  'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                  'sub_submit_success', 'sub_car_request_submit_click']
    df_hits.loc[(df_hits.event_action.isin(list_of_ca) == True), 'cr'] = 1
    df_hits.loc[(df_hits.event_action.isin(list_of_ca) == False), 'cr'] = 0
    df_cr_grouped = df_hits.groupby(['session_id']).agg({'cr': max})
    df_sessions = pd.merge(left=df_sessions, right=df_cr_grouped, on='session_id', how='inner')

    x = df_sessions.drop('cr', axis=1)
    y = df_sessions['cr']

    preprocessor = Pipeline(steps=[
        ('new_features', FunctionTransformer(new_features)),
        ('filter', FunctionTransformer(drop_columns))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    transformation = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'int32', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ])

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('transformation', transformation),
        ('classifier', DecisionTreeClassifier())
    ])
    pipe.fit(x, y)
    score = roc_auc_score(y, pipe.predict_proba(x)[:, 1])
    print(f'model: {type(pipe.named_steps["classifier"]).__name__}, roc_auc_score: {score:.4f}')

    with open('model/Final_project_model.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Final_project',
                'author': 'Bogomaz Ekaterina',
                'type': type(pipe.named_steps["classifier"]).__name__,
                'roc_auc_score': score
            }
        }, file)


if __name__ == '__main__':
    main()
