import streamlit as st
import pandas as pd
import sklearn
import tensorflow as tf
import warnings
from tensorflow import keras
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import cv2
import pickle
import gc
import seaborn as sns
from utils import custom_imputer, outlier_removal, replace_add_drop_num, replace_add_drop_cat
st.set_option('deprecation.showPyplotGlobalUse', False)

gc.enable()


def load_tuned_models():
    tuned_names = ['Random Forest tuned', 'CatBoost tuned', 'XGBoost tuned']
    tuned_names_load = ['rf_tuned', 'cat_tuned', 'xgb_tuned']
    names = dict(zip(tuned_names, tuned_names_load))
    models = {}
    for name, load_name in names.items():
        models[name] = pickle.load(open('streamlit/models/' + load_name + '.pkl', 'rb'))
    models['Neural Network'] = keras.models.load_model('streamlit/models/nn_model.h5')

    return models


def load_data_raw():
    df_raw = pd.read_csv('streamlit/train.csv')
    return df_raw


def preprocess(df_raw):
    outlier_remover = FunctionTransformer(outlier_removal, kw_args={'col':'Item_Visibility', 'factor':1.5})

    one_hot_cols = ['New_Item_Type', 'Item_Fat_Content', 'Outlet_Identifier', 'Outlet_Size',
                    'Outlet_Location_Type', 'Outlet_Type']
    scale_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year', 'item_visible_avg']

    col_trans_pipe = ColumnTransformer([
        ('scale', StandardScaler(), scale_cols),
        ('one_hot', OneHotEncoder(), one_hot_cols)
    ])

    prep_pipe_full = Pipeline([
        ('custom_imputer', custom_imputer()),
        ('replace_add_drop_cat', FunctionTransformer(replace_add_drop_cat)),
        ('replace_add_drop_num', replace_add_drop_num()),
        ('col_trans_pipe', col_trans_pipe)
    ])

    df_raw_no_out = outlier_remover.fit_transform(df_raw)
    X = df_raw_no_out.drop('Item_Outlet_Sales', axis=1)
    y = df_raw_no_out['Item_Outlet_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=120)

    X_train = prep_pipe_full.fit_transform(X_train)
    X_test = prep_pipe_full.transform(X_test)

    del X, y, df_raw, df_raw_no_out

    return X_train, y_train, X_test, y_test, col_trans_pipe, prep_pipe_full

def train(df_raw):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(n_jobs=-1),
        'KNeighborsRegressor': KNeighborsRegressor(n_jobs=-1),
        'Support Vector Regressor': SVR(kernel='linear'),
        'XGBoost': XGBRegressor(n_jobs=-1),
        'CatBoost': CatBoostRegressor(),
        'ExtraTrees': ExtraTreesRegressor(n_jobs=-1)
    }

    trained_models = {}

    X_train, y_train, X_test, y_test, col_trans_pipe, _ = preprocess(df_raw)

    for name, algo in models.items():
        model = algo.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models, X_test, y_test, X_train, y_train, col_trans_pipe

def predict(trained_models, tuned_models, X_test, y_test, X_train, y_train):
    trained_models.update(tuned_models)
    r2_test = [round(r2_score(y_test, model.predict(X_test)) * 100, 2) for model in list(trained_models.values())]
    mse_test = [round(mean_squared_error(y_test, model.predict(X_test))) for model in list(trained_models.values())]
    mae_test = [round(mean_absolute_error(y_test, model.predict(X_test)), 2) for model in list(trained_models.values())]
    names = [name for name in list(trained_models.keys())]
    models_performance_test = {'r2': r2_test, 'mse': mse_test, 'mae': mae_test}
    models_performance_test = pd.DataFrame(models_performance_test, index=names).sort_values(by=['r2'])

    r2_train = [round(r2_score(y_train, model.predict(X_train)) * 100, 2) for model in list(trained_models.values())]
    mse_train = [round(mean_squared_error(y_train, model.predict(X_train))) for model in list(trained_models.values())]
    mae_train = [round(mean_absolute_error(y_train, model.predict(X_train)), 2) for model in list(trained_models.values())]
    models_performance_train = {'r2': r2_train, 'mse': mse_train, 'mae': mae_train}
    models_performance_train = pd.DataFrame(models_performance_train, index=names).sort_values(by=['r2'])

    del trained_models, tuned_models, X_test, X_train, y_test, y_train

    return models_performance_test, models_performance_train

def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """

    # Remove the internal helper function
    # check_is_fitted(column_transformer)

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                          "provide get_feature_names. "
                          "Will return input column names if available"
                          % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [f for f in column]

        return [f for f in trans.get_feature_names()]

    ### Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names




def main_section():
    st.title('Sales Prediction Project')
    background_im = cv2.imread('streamlit/images/background.jpeg')
    st.image(cv2.cvtColor(background_im, cv2.COLOR_BGR2RGB), use_column_width=True)
    st.markdown('**Visualisation and EDA** section contains some plots and graphs as well as some basic '
            'information based on the raw data. **Model Selection and Performance** section provides information '
                'about tested models and their relative performance. In the **Feature Importances** section importance of the '
                'top 10 features predicted by different models is shown. In the **Prediction Service** section the best model'
                ' is deployed and one can make predictions based on the manually typed data.')
    del background_im
    gc.collect()

def vis_and_eda():
    st.title('Visualization and EDA')
    df_raw = load_data_raw()
    st.success('Data successfully loaded')
    if st.sidebar.checkbox('Display shape'):
        st.write('Size of the raw data: ', df_raw.shape)
    if st.sidebar.checkbox('Display summary'):
        st.write(df_raw.describe())
    if st.sidebar.checkbox('Display null values'):
        st.write(df_raw.isnull().sum())
    if st.sidebar.checkbox('Display data types'):
        st.write(df_raw.dtypes)
    if st.sidebar.checkbox('Display heatmap'):
        fig, ax = plt.subplots()
        sns.heatmap(df_raw.corr(), annot=True, ax=ax)
        st.pyplot(fig)
    if st.sidebar.checkbox('Display pairplot'):
        st.write(sns.pairplot(df_raw, diag_kind='kde'))
        st.pyplot()
    if st.sidebar.checkbox('Display distributions'):
        for i in df_raw.describe().columns:
            fig, ax = plt.subplots()
            sns.displot(df_raw[i], kde=True)
            st.pyplot()
    if st.sidebar.checkbox('Display boxplots'):
        for i in df_raw.describe().columns:
            fig, ax = plt.subplots()
            sns.boxplot(df_raw[i])
            st.pyplot()
    if st.sidebar.checkbox('Display countplots'):
        for i in df_raw.select_dtypes(include='object').columns[1:]:
            fig, ax = plt.subplots()
            chart = sns.countplot(df_raw[i])
            chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
            st.pyplot()
    if st.sidebar.checkbox('Display scatterplot'):
        fig, ax = plt.subplots(figsize=(15,15))
        sns.scatterplot(x='Item_Visibility', y='Item_Outlet_Sales', hue='Item_Type', size='Item_Weight', data=df_raw)
        st.pyplot()

    del df_raw
    gc.collect()

def model_selection_and_performance():
    st.title('Model Selection and Performance')
    df_raw = load_data_raw()
    trained_models, X_test, y_test, X_train, y_train, _ = train(df_raw)
    tuned_models = load_tuned_models()
    models_performance_test, models_performance_train = predict(trained_models, tuned_models, X_test, y_test, X_train,
                                                                y_train)
    option = st.sidebar.selectbox('Select Data', ['Train Data Performance', 'Test Data Performance'])
    if option == 'Train Data Performance':
        fig, ax = plt.subplots()
        chart = sns.barplot(x=models_performance_train.index, y='r2', data=models_performance_train)
        for p in chart.patches:
            chart.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           size=8,
                           xytext=(0, -12),
                           textcoords='offset points')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        st.pyplot()

        fig, ax = plt.subplots()
        chart = sns.barplot(x=models_performance_train.index, y='mae', data=models_performance_train)
        for p in chart.patches:
            chart.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           size=8,
                           xytext=(0, -12),
                           textcoords='offset points')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        st.pyplot()

        st.dataframe(models_performance_train)

    if option == 'Test Data Performance':
        fig, ax = plt.subplots()
        chart = sns.barplot(x=models_performance_test.index, y='r2', data=models_performance_test)
        for p in chart.patches:
            chart.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           size=8,
                           xytext=(0, -12),
                           textcoords='offset points')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        st.pyplot()

        fig, ax = plt.subplots()
        chart = sns.barplot(x=models_performance_test.index, y='mae', data=models_performance_test)
        for p in chart.patches:
            chart.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           size=8,
                           xytext=(0, -12),
                           textcoords='offset points')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        st.pyplot()

        st.dataframe(models_performance_test)

    del df_raw, trained_models, tuned_models, X_test, X_train, y_test, y_train, models_performance_train, models_performance_test
    gc.collect()

def feature_importances():
    models = load_tuned_models()
    df_raw = load_data_raw()
    X_train, y_train, X_test, y_test, col_trans_pipe, _ = preprocess(df_raw)
    importances = pd.DataFrame({
        'Feature': get_feature_names(col_trans_pipe),
        'Importance_RF': models['Random Forest tuned'].best_estimator_.feature_importances_,
        'Importance_CatBoost': models['CatBoost tuned'].best_estimator_.feature_importances_ / 100,
        'Importance_XGB': models['XGBoost tuned'].feature_importances_
    })
    importances = importances.set_index('Feature')

    fig, ax = plt.subplots()
    importances['Importance_RF'].sort_values(ascending=False)[:10].plot(kind='bar',
                                                                        title='Feature Importances Random Forest', color='r', ax=ax)
    st.pyplot()

    fig, ax = plt.subplots()
    importances['Importance_CatBoost'].sort_values(ascending=False)[:10].plot(kind='bar',
                                                                              title='Feature Importances CatBoost', ax=ax)
    st.pyplot()

    fig, ax = plt.subplots()
    importances['Importance_XGB'].sort_values(ascending=False)[:10].plot(kind='bar',
                                                                         title='Feature Importances XGB', color='g', ax=ax)
    st.pyplot()

    gc.collect()

def prediction_service():
    df_raw = load_data_raw()
    features = []
    features.append(st.text_input('Item Identifier', 0))
    features.append(float(st.text_input('Item Weight', 0)))
    features.append(st.selectbox('Select Item Fat Content', df_raw['Item_Fat_Content'].unique()))
    features.append(float(st.text_input('Item Visibility', 0)))
    features.append(st.selectbox('Select Item Type', df_raw['Item_Type'].unique()))
    features.append(float(st.text_input('Item MRP', 0)))
    features.append(st.selectbox('Select Outlet Identifier', df_raw['Outlet_Identifier'].unique()))
    features.append(int(st.text_input('Outlet Establishment Year', 0)))
    features.append(st.selectbox('Select Outlet Size', df_raw['Outlet_Size'].unique()))
    features.append(st.selectbox('Select Outlet Location Type', df_raw['Outlet_Location_Type'].unique()))
    features.append(st.selectbox('Select Outlet Type', df_raw['Outlet_Type'].unique()))
    if st.button('Predict'):
        df = pd.DataFrame([features], columns=df_raw.columns[:-1])
        models = load_tuned_models()
        X_train, y_train, X_test, y_test, col_trans_pipe, prep_pipe_full = preprocess(df_raw)
        del X_train, X_test, y_train, y_test, col_trans_pipe
        df_prep = prep_pipe_full.transform(df)
        prediction = round(models['CatBoost tuned'].predict(df_prep)[0], 4)
        st.info('Predicted Item Outlet Sales: {}'.format(prediction))
        del df_raw, df_prep, df
    else:
        st.info('Enter all values')

activities = ['Main', 'Visualization and EDA', 'Model Selection and Performance', 'Feature Importances', 'Prediction Service', 'About']
option = st.sidebar.selectbox('Select Option', activities)

if option == 'Main':
    main_section()

if option == 'Visualization and EDA':
    vis_and_eda()
    gc.collect()

if option == 'Model Selection and Performance':
    model_selection_and_performance()
    gc.collect()

if option == 'Feature Importances':
    feature_importances()
    gc.collect()

if option == 'Prediction Service':
    prediction_service()
    gc.collect()

if option == 'About':
    st.title('About')
    st.write('This is an interactive website for the Sales Prediction Project. Data was taken from Udemy ML course.')


