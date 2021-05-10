import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from utils import custom_imputer, replace_add_drop_cat, replace_add_drop_num

app = Flask(__name__)
model = pickle.load(open('cat_tuned.pkl', 'rb'))
pipe = pickle.load(open('prep_pipe_full.pkl', 'rb'))
col_names = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    output = [x for x in request.form.values()]
    if output[-3] in ['NaN', 'nan', 'NAN']:
        output[-3] = np.NaN
    output[1] = float(output[1])
    output[3] = float(output[3])
    output[5] = float(output[5])
    output[7] = int(output[7])
    df = pd.DataFrame([output], columns=col_names)
    df_prep = pipe.transform(df)
    prediction = round(model.predict(df_prep)[0], 4)
    return render_template('index.html', prediction_text='Predicted Item Outlet Sales: {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)