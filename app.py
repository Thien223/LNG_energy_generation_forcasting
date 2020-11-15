import pickle
from flask import Flask, render_template, request, json, redirect
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
### training prediction model using linear regression

UPLOAD_FOLDER = 'static/data/'
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def read_config():
    ### read csv file
    config = pd.read_csv('static/data/inlet_clean_info.csv')
    ### separate each variable
    inlet_code = config['base'][0] ### "base" column, row 0
    cleaned_time = config['cleaned_time'][0] ### "cleaned_time" column, row 0
    tested_power_output = config['tested_power_output'][0] ### "tested_power_output" column, row 0
    ###return values
    return {'inlet_code':inlet_code,'cleaned_time':cleaned_time, 'tested_power_output':tested_power_output}


# csv_path = 'data/input_template.csv'
# startTime, endTime = '2019-05-03 16:00:00','2019-05-03 18:00:00'

def get_input_from_csv(csv_path, startTime, endTime):
    input_data_ = pd.read_csv(csv_path)
    input_data_['Time'] = pd.to_datetime(input_data_['Time'])
    input_data_.set_index('Time', inplace=True)
    input_data = input_data_[startTime:endTime].copy()
    return input_data


def update_model():
    import numpy as np
    data = pd.read_csv('d:\\boston_housing.csv')
    corr = data.corr()
    import matplotlib.pyplot as plt
    plt.imshow(corr)
    split = int(len(data)*0.8)

    train_data = data[:split].copy()
    train_data = train_data.sort_values(by='clean_time')
    test_data = data[split:].copy()


    x_train = train_data['clean_time'].copy()
    y_train = train_data['reduction'].copy()
    x_test = test_data['clean_time'].copy()
    y_test = test_data['reduction'].copy()
    error = []
    model=[]
    for degree in range(1,16):
        transformer = PolynomialFeatures(degree=degree)
        ## transform input
        X_poly = transformer.fit_transform(np.array(x_train).reshape(-1,1))
        ## create model
        model_ = LinearRegression()
        ## fit model
        model_.fit(X_poly, y_train)
        ### write model to model.pkl

        X_poly = transformer.fit_transform(np.array(x_test).reshape(-1,1))
        pred_y = model_.predict(X=X_poly)
        ### store models
        model.append(model_)
        ### store error rate and pick the good one
        error_ = []
        for pred, real in zip(pred_y,y_test):
            e = ((abs(pred - real) / real) * 100)
            error_.append(e)

        error.append(np.mean(error_)**2 + np.std(error_)**2)
    idx = error.index(min(error))
    transformer = PolynomialFeatures(degree=idx+1) ### +1 because the index start from 0, while our degree start from 1

    pickle.dump(model[idx], open('model.pkl','wb'))
    pickle.dump(transformer, open('transformer.pkl','wb'))
    return


@app.route('/inference', methods=['POST'])
def inference():
    import datetime
    import pandas as pd
    update_model()
    ### get input data from request
    date = request.form['date-select']
    pressure = float(request.form['txt-pressure'])
    #todo: do we really need this variable?
    cleaned_operated_time = float(request.form['txt-cleaned-time']) - 6
    temp_suffix=['-15','-18','-21','00','03','06','09','12','15','18','21','-00','-03','-06']
    temp_list =[]
    cleaned_time_list = []

    for suffix in temp_suffix:
        if suffix[0]=='-' and int(suffix[1:])>=15:
            temp_list.append(float(request.form['input-temp-' + suffix]))
            d = pd.to_datetime(date).date() - datetime.timedelta(days=1)
            d = pd.to_datetime(d).replace(hour=int(suffix[1:]))
            cleaned_time_list.append(d)
        elif suffix[0]=='-' and int(suffix[1:])<=6:
            temp_list.append(float(request.form['input-temp-' + suffix]))
            d = pd.to_datetime(date).date() + datetime.timedelta(days=1)
            d = pd.to_datetime(d).replace(hour=int(suffix[1:]))
            cleaned_time_list.append(d)
        else:
            temp_list.append(float(request.form['input-temp-' + suffix]))
            d = pd.to_datetime(date).replace(hour=int(suffix))
            cleaned_time_list.append(d)
    #
    cleaned_time_df = pd.to_datetime(cleaned_time_list)
    temp_df = pd.DataFrame(temp_list, index= cleaned_time_df)
    ##interpolating time to hour frequency
    temp_df_ = temp_df[1:-1].resample('H').interpolate()

    config=read_config()

    EBH_list=[cleaned_operated_time + i for i in range(34)]
    # EBH_list = [(item if item<=2250 else 2250)  for item in EBH_list]
    ### CV
    cv = 415 - config['tested_power_output']
    delta_temps = [temp - 15.0 for temp in temp_df_.iloc[:,0]]
    alpha_1 = [(2.655975/100000000)  * delta_temp**4 + (1.375295/1000000) * delta_temp**3 + (4.191724/100000)*delta_temp**2 + (4.861757/1000) * delta_temp + 1 for delta_temp in delta_temps]

    delta_pressure = pressure - 1013
    alpha_2 = (1.051076/1000000)  * delta_pressure**2 - (9.850922/10000) * delta_pressure + 1

    alpha_3 = [(- 0.00251*ebh**2 + 10.69784*ebh - 6.60637) for ebh in EBH_list]
    #
    # print('alpha 1 2 3')
    # print(len(alpha_1))
    # print(alpha_2)
    # print(len(alpha_3))
    cc_generate_predicted = [(415 / (alpha_1_ * alpha_2)) - alpha_3_/1000 - cv for alpha_1_,alpha_3_ in zip(alpha_1, alpha_3)]
    gt_generate_predicted = [(277 / (alpha_1_ / alpha_2)) - alpha_3_/1000 - cv  for alpha_1_,alpha_3_ in zip(alpha_1, alpha_3)]
    cc_transfer_predicted = [cc_generate_predicted_ - 6 for cc_generate_predicted_ in cc_generate_predicted]
    gt_transfer_predicted = [gt_generate_predicted_ - 6 for gt_generate_predicted_ in gt_generate_predicted]
    return json.dumps({'power_reduction': list(alpha_3),'cc_generate_predicted':cc_generate_predicted,'gt_generate_predicted':gt_generate_predicted,'cc_transfer_predicted':cc_transfer_predicted,'gt_transfer_predicted':gt_transfer_predicted,'temperature':list(temp_df_.iloc[:,0]), 'pressure':pressure})



@app.route('/ml-inference', methods=['POST'])
def ml_inference():
    import datetime
    import pandas as pd
    update_model()
    ### get input data from request
    date = request.form['date-select']
    pressure = float(request.form['txt-pressure'])
    #todo: do we really need this variable?
    cleaned_operated_time = float(request.form['txt-cleaned-time']) -6
    temp_suffix=['-15','-18','-21','00','03','06','09','12','15','18','21','-00','-03','-06']
    temp_list =[]
    cleaned_time_list = []

    for suffix in temp_suffix:

        if suffix[0]=='-' and int(suffix[1:])>=15:
            temp_list.append(float(request.form['input-temp-' + suffix]))
            d = pd.to_datetime(date).date() - datetime.timedelta(days=1)
            d = pd.to_datetime(d).replace(hour=int(suffix[1:]))
            cleaned_time_list.append(d)
            # print(float(request.form['input-temp-' + suffix]))
        elif suffix[0]=='-' and int(suffix[1:])<=6:
            temp_list.append(float(request.form['input-temp-' + suffix]))
            d = pd.to_datetime(date).date() + datetime.timedelta(days=1)
            d = pd.to_datetime(d).replace(hour=int(suffix[1:]))
            cleaned_time_list.append(d)
            # print(float(request.form['input-temp-' + suffix]))
        else:
            temp_list.append(float(request.form['input-temp-' + suffix]))
            d = pd.to_datetime(date).replace(hour=int(suffix))
            cleaned_time_list.append(d)
            # print(float(request.form['input-temp-' + suffix]))

    #
    cleaned_time_df = pd.to_datetime(cleaned_time_list)
    temp_df = pd.DataFrame(temp_list, index= cleaned_time_df)
    ##interpolating time to hour frequency
    temp_df_ = temp_df[1:-1].resample('H').interpolate()

    config=read_config()

    EBH_list=[cleaned_operated_time + i for i in range(34)]
    EBH_list = [(item if item<=2250 else 2250)  for item in EBH_list]
    ####load predict model and calculate reduction of power
    model = pickle.load(open('model.pkl', 'rb'))
    transformer = pickle.load(open('transformer.pkl', 'rb'))
    EBH_poly = transformer.fit_transform(np.array(EBH_list).reshape(-1,1))
    power_reduction = model.predict(EBH_poly)

    cv = 415 - config['tested_power_output']
    delta_temps = [temp - 15.0 for temp in temp_df_.iloc[:,0]]
    alpha_1 = [(2.655975/100000000)  * delta_temp**4 + (1.375295/1000000) * delta_temp**3 + (4.191724/100000)*delta_temp**2 + (4.861757/1000) * delta_temp + 1 for delta_temp in delta_temps]

    delta_pressure = pressure - 1013
    alpha_2 = (1.051076/1000000)  * delta_pressure**2 - (9.850922/10000) * delta_pressure + 1

    alpha_3 = power_reduction

    cc_generate_predicted = [(415 / (alpha_1_ * alpha_2)) - alpha_3_/1000 - cv for alpha_1_,alpha_3_ in zip(alpha_1, alpha_3)]
    gt_generate_predicted = [(277 / (alpha_1_ / alpha_2)) - alpha_3_/1000 - cv  for alpha_1_,alpha_3_ in zip(alpha_1, alpha_3)]
    cc_transfer_predicted = [cc_generate_predicted_ - 6 for cc_generate_predicted_ in cc_generate_predicted]
    gt_transfer_predicted = [gt_generate_predicted_ - 6 for gt_generate_predicted_ in gt_generate_predicted]

    return json.dumps({'power_reduction': list(alpha_3),'cc_generate_predicted':cc_generate_predicted,'gt_generate_predicted':gt_generate_predicted,'cc_transfer_predicted':cc_transfer_predicted,'gt_transfer_predicted':gt_transfer_predicted,'temperature':list(temp_df_.iloc[:,0]), 'pressure':pressure})



@app.route('/schedule-ml-inference', methods=['POST'])
def schedule_ml_inference():
    import datetime
    import pandas as pd
    update_model()
    ### get input data from request
    date = request.form['date-select']
    pressure = float(request.form['txt-pressure'])
    #todo: do we really need this variable?
    cleaned_operated_time = float(request.form['txt-cleaned-time']) - 6
    temp_suffix=['-15','-18','-21','00','03','06','09','12','15','18','21','-00','-03','-06']
    temp_list =[]
    temp_list_=[]
    cleaned_time_list = []

    for suffix in temp_suffix:
        temp_list.append(float(request.form['input-temp-' + suffix]))
        if suffix[0] == '-' and int(suffix[1:]) >= 15:
            d = pd.to_datetime(date).date() - datetime.timedelta(days=1)
            d = pd.to_datetime(d).replace(hour=int(suffix.replace('-','')))
            cleaned_time_list.append(d)
        elif suffix[0] == '-' and int(suffix[1:]) <= 6:
            d = pd.to_datetime(date).date() + datetime.timedelta(days=1)
            d = pd.to_datetime(d).replace(hour=int(suffix[1:]))
            cleaned_time_list.append(d)
        else:
            d = pd.to_datetime(date).replace(hour=int(suffix))
            cleaned_time_list.append(d)


    for i in range(6):
        temp_list_.append(temp_list[1])
    # temp_list_.append(temp_list[1])
    # temp_list_.append(temp_list[1]+(temp_list[1]-temp_list[2])*1/3)
    # temp_list_.append(temp_list[1]+(temp_list[1]-temp_list[2])*2/3)
    # temp_list_.append(temp_list[2])
    # temp_list_.append(temp_list[2]+(temp_list[2]-temp_list[3])*1/3)
    # temp_list_.append(temp_list[2]+(temp_list[2]-temp_list[3])*2/3)

    for i in range(8):
        temp_list_.append(temp_list[3])

    for i in range(3):
        temp_list_.append(temp_list[6])
    for i in range(7):
        temp_list_.append(temp_list[7])


    for i in range(6):
        temp_list_.append(temp_list[10])

    for i in range(4):
        temp_list_.append(temp_list[11])
    #

    cleaned_time_list = pd.date_range(cleaned_time_list[0], cleaned_time_list[-1], freq='H')

    ### get system info (cleaned time, tested output)
    ##{'inlet_code':inlet_code,'cleaned_time':cleaned_time, 'tested_power_output':tested_power_output}
    config=read_config()

    temp_df_ = pd.DataFrame(temp_list_, index=cleaned_time_list[3:-3])

    EBH_list=[cleaned_operated_time + i for i in range(34)]
    EBH_list = [(item if item<=2250 else 2250)  for item in EBH_list]
    ####load predict model and calculate reduction of power
    model = pickle.load(open('model.pkl', 'rb'))
    transformer = pickle.load(open('transformer.pkl', 'rb'))
    EBH_poly = transformer.fit_transform(np.array(EBH_list).reshape(-1,1))
    power_reduction = model.predict(EBH_poly)

    ### CV
    cv = 415 - config['tested_power_output']
    delta_temps = [temp - 15.0 for temp in temp_df_.iloc[:,0]]
    alpha_1 = [(2.655975/100000000)  * delta_temp**4 + (1.375295/1000000) * delta_temp**3 + (4.191724/100000)*delta_temp**2 + (4.861757/1000) * delta_temp + 1 for delta_temp in delta_temps]

    delta_pressure = pressure - 1013
    alpha_2 = (1.051076/1000000)  * delta_pressure**2 - (9.850922/10000) * delta_pressure + 1

    alpha_3 = power_reduction

    cc_generate_predicted = [(415 / (alpha_1_ * alpha_2)) - alpha_3_/1000 - cv for alpha_1_,alpha_3_ in zip(alpha_1, alpha_3)]
    gt_generate_predicted = [(277 / (alpha_1_ / alpha_2)) - alpha_3_/1000 - cv  for alpha_1_,alpha_3_ in zip(alpha_1, alpha_3)]
    cc_transfer_predicted = [cc_generate_predicted_ - 6 for cc_generate_predicted_ in cc_generate_predicted]
    gt_transfer_predicted = [gt_generate_predicted_ - 6 for gt_generate_predicted_ in gt_generate_predicted]

    return json.dumps({'power_reduction': list(alpha_3),'cc_generate_predicted':cc_generate_predicted,'gt_generate_predicted':gt_generate_predicted,'cc_transfer_predicted':cc_transfer_predicted,'gt_transfer_predicted':gt_transfer_predicted,'temperature':list(temp_df_.iloc[:,0]), 'pressure':pressure})



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploadss', methods = ['GET', 'POST'])
def upload():
    import os
    print(request.files['file'])
    if request.method=='POST':
        file = request.files['file']
        filenames = ['temperature.csv', 'inlet_clean_info.csv', 'log_data.csv', 'medium-term-temperature.csv']
        if file.filename in filenames:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return 'file uploaded successfully'
        else:
            return 'filename is not acceptable'

if __name__ == "__main__":
    print(("Starting server..."
        "please wait until server has fully started"))
    app.run(host='0.0.0.0', port='80',threaded=True)

