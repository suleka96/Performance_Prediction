import csv
import warnings
import math
import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import xgboost as xg
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle


matplotlib.use('agg')
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_style("whitegrid")


def get_data(path):
    dataset = pandas.read_csv(path)

    if path == "Datasets/APIM_Dataset.csv":
        dataset = dataset.loc[dataset["Error %"] < 5]
        dataset['Name'] = pandas.Categorical(dataset['Name']).codes

        _X = dataset[['Name', 'Concurrent Users', 'Message Size (Bytes)', 'Sleep Time (ms)']]
        _y_avg = dataset['Average (ms)']
        _y_throughput = dataset['Throughput']

        seed = 42
        x_, y_avg_, y_throughput_ = scale_data(seed, _X, _y_avg, _y_throughput)
        return x_, y_avg_, y_throughput_

    elif path == "Datasets/Ballerina_Dataset.csv":
        dataset['Name'] = pandas.Categorical(dataset['Name']).codes
        dataset = dataset.loc[dataset["Error %"] < 5]

        _X = dataset[['Name', 'Concurrent Users', 'Message Size (Bytes)', 'Sleep Time (ms)']]
        _y_avg = dataset['Average (ms)']
        _y_throughput = dataset['Throughput']
        seed = 65
        x_, y_avg_, y_throughput_ = scale_data(seed, _X, _y_avg, _y_throughput)

        return x_, y_avg_, y_throughput_

    elif path == "Datasets/Springboot_Dataset.csv":
        # Filter Data
        dataset = dataset.loc[dataset["error_rate"] < 5]

        # Convert Heap
        dataset['heap'] = pandas.to_numeric(dataset['heap'].str.replace(r'[a-z]+', ''), errors='coerce')

        # Convert Instance Type and Proxy values to numeric
        dataset['use case'] = pandas.Categorical(dataset['use case']).codes
        dataset['collector'] = pandas.Categorical(dataset['collector']).codes

        # Define features and dependants
        _X = dataset[['use case', 'size', 'user', 'heap', 'collector']]
        _y_avg = dataset['average_latency']
        _y_throughput = dataset['throughput']
        seed = 42
        x_, y_avg_, y_throughput_ = scale_data(seed, _X, _y_avg, _y_throughput)
        return x_, y_avg_, y_throughput_

    elif path == "Datasets/tpcw.csv":
        _X = dataset[['concurrent_users', 'cores', 'workload_mix']]
        _y_avg = dataset['latency']
        _y_throughput = dataset['tps']
        seed = 42
        x_, y_avg_, y_throughput_ = scale_data(seed, _X, _y_avg, _y_throughput)

        return x_, y_avg_, y_throughput_


def scale_data(seed, _X, y_avg_, y_throughput_):
    scaler = MinMaxScaler(feature_range=(0, 1))
    _X = scaler.fit_transform(_X)

    _X, _y_avg, _y_throughput = shuffle(_X, y_avg_, y_throughput_, random_state=seed)
    return _X, _y_avg, _y_throughput


# Models -----------------------------------------------------------------------------------------------------------
def svr_linear_kernel(name,X, y):
    print("svr_linear_kernel")
    gsc = GridSearchCV(
        estimator=SVR(kernel='linear'),
        param_grid={
            'C': [0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_

    print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    linear_svr = SVR(kernel='linear', C=best_params["C"], epsilon=best_params["epsilon"], coef0=0.1, shrinking=True,
                   tol=0.001, cache_size=200, verbose=False, max_iter=-1)

    predicted = get_predictions(name,linear_svr, X, y)
    return predicted


def svr_poly_kernel(name,X, y):
    print("svr_poly_kernel")
    gsc = GridSearchCV(
        estimator=SVR(kernel='poly'),
        param_grid={
            'C': [0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'degree': [2, 3, 4],
            'coef0': [0.1, 0.01, 0.001, 0.0001]

        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_

    print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    poly_svr = SVR(kernel='poly', C=best_params["C"], epsilon=best_params["epsilon"], coef0=best_params["coef0"],
                   degree=best_params["degree"], shrinking=True,
                   tol=0.001, cache_size=200, verbose=False, max_iter=-1)

    predicted = get_predictions(name,poly_svr, X, y)
    return predicted


def svr_rbf_kernel(name,X, y):
    print("svr_rbf_kernel")
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'gamma':[0.0001, 0.001, 0.002, 0.005, 0.1, 0.2]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_

    print('\nBest: %f using %s' % (grid_result.best_score_, grid_result.best_params_))

    rbf_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"], coef0=0.1, shrinking=True,
                   tol=0.001, cache_size=200, verbose=False, max_iter=-1)

    predicted = get_predictions(name,rbf_svr, X, y)
    return predicted


def xgb_model(name,X, y):
    print("xgb_model")
    gsc = GridSearchCV(
        estimator=xg.XGBRegressor(),
        param_grid={
          'learning_rate': (0.06, 0.08),
          'max_depth': (5, 7),
          'subsample': (0.5, 0.75),
          'colsample_bytree': (0.5, 1),
          'n_estimators': (10, 50, 100, 1000)
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_

    print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    xgb = xg.XGBRegressor(learning_rate=best_params["learning_rate"],
                          max_depth=best_params["max_depth"], subsample=best_params["subsample"],
                          colsample_bytree= best_params["colsample_bytree"], n_estimators=best_params["n_estimators"],
                          coef0=0.1, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)

    predicted = get_predictions(name,xgb, X, y)
    return predicted


def rfr_model(name,X, y):
    print("rfr_model")
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'random_state': (False, False),
            'max_depth': (3, 5, 7),
            'n_estimators': (100, 1000),
            # 'min_samples_split': (2, 5, 7),
            # 'min_samples_leaf': (10, 50)
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_

    print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
                                random_state=best_params["random_state"],
                                # min_samples_leaf=best_params["min_samples_leaf"], min_samples_split=best_params["min_samples_split"],
                                verbose=False)

    predicted = get_predictions(name,rfr, X, y)
    return predicted


def get_predictions(name,model, X, y):
    scoring = {'r_score': 'r2',
               'abs_error': 'neg_mean_absolute_error',
               'squared_error': 'neg_mean_squared_error'}

    predictions = cross_val_predict(model, X, y, cv=10)
    scores = cross_validate(model, X, y, cv=10, scoring=scoring, return_train_score=True)

    mse_val = abs(scores['test_squared_error'].mean())
    RMSE = math.sqrt(mse_val)
    MAE = abs(scores['test_abs_error'].mean())
    MAPE = mean_absolute_percentage_error(y, predictions)
    print("Scores\n"
          "RMSE :", math.sqrt(mse_val),
          "| MAPE: ", MAPE,
          )

    fileName = "datafiles/"+name+".csv"
    with open(fileName, "a") as f:
        writer = csv.writer(f)
        writer.writerows(zip(y,predictions))

    with open("results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerows(zip([name], [RMSE],[MAE],[MAPE]))

    return predictions


def mean_absolute_percentage_error(y_true, prediction):
    y_true, y_pred = np.array(y_true), np.array(prediction)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


# Plot -----------------------------------------------------------------
def make_diagonal(x, y):
    xx, yy = [min(x), max(y)], [min(x), max(y)]
    plt.plot(xx, yy)


def label_plots(label):
    if label == 'y_avg':
        pylab.xlabel('Measured Average Latency(ms)')
        pylab.ylabel('Predicted Average Latency(ms)')
    else:
        pylab.xlabel('Measured Throughput(Requests/Second)')
        pylab.ylabel('Predicted Throughput(Requests/Second)')


def graph(name,dependent, label, linear_pred, poly_pred, rbf_pred, rfr_pred, xgb_pred,bayesPred):
# def graph(name, dependent, label,xgb_pred):

    fig, ax = plt.subplots()
    fig.set_size_inches(15,10)

    pylab.subplot(231)
    pylab.scatter(dependent, linear_pred, color='darkorange')
    pylab.title('SVR (Linear)')
    label_plots(label)
    make_diagonal(dependent, linear_pred)

    pylab.subplot(232)
    pylab.scatter(dependent, poly_pred, color='blue')
    make_diagonal(dependent, poly_pred)
    label_plots(label)
    pylab.title('SVR (Poly)')

    pylab.subplot(233)
    pylab.scatter(dependent, rbf_pred, color='g')
    make_diagonal(dependent, rbf_pred)
    label_plots(label)
    pylab.title('SVR (RBF)')

    pylab.subplot(234)
    pylab.scatter(dependent, rfr_pred, color='purple')
    make_diagonal(dependent, rfr_pred)
    label_plots(label)
    pylab.title('Random Forest Regression')

    pylab.subplot(235)
    pylab.scatter(dependent, xgb_pred, color='r')
    make_diagonal(dependent, xgb_pred)
    label_plots(label)
    pylab.title('XGBoost')

    pylab.subplot(236)
    pylab.scatter(dependent, bayesPred, color='brown')
    make_diagonal(dependent, bayesPred)
    label_plots(label)
    pylab.title('Bayesian')

    plt.tight_layout()

    ## pylab.subplots_adjust(top=1, left=0.3)


    pylab.show()
    plt.savefig(name, format='png',  transparent=False)
    plt.close()


if __name__ == '__main__':
    # file_path = "Datasets/APIM_Dataset.csv"
    # file_path = "Datasets/Ballerina_Dataset.csv"
    file_path = "Datasets/Springboot_Dataset.csv"
    # file_path = "Datasets/tpcw.csv"

    X, y_avg, y_throughput = get_data(file_path)


    # Average
    linear_pred = svr_linear_kernel("average_linear_pred",X, y_avg)
    poly_pred = svr_poly_kernel("average_poly_pred",X, y_avg)
    rbf_pred=svr_rbf_kernel("average_rbf_pred",X, y_avg)
    rfr_pred=rfr_model("average_rfr_pred",X, y_avg)
    xgb_pred=xgb_model("average_xgb_pred",X, y_avg)
    import BayesianModel
    bayesPred = BayesianModel.runBayesian("average_bayesPred","avg",0.03,file_path)
    graph("avg.png",y_avg, "y_avg", linear_pred, poly_pred, rbf_pred, rfr_pred, xgb_pred,bayesPred)
    # graph("avg.png",y_avg, "y_avg", xgb_pred)

    # Throughput
    linear_pred=svr_linear_kernel("throughput_linear_pred",X, y_throughput)
    poly_pred=svr_poly_kernel("throughput_poly_pred",X, y_throughput)
    rbf_pred=svr_rbf_kernel("throughput_rbf_pred",X, y_throughput)
    rfr_pred=rfr_model("throughput_rfr_pred",X, y_throughput)
    xgb_pred=xgb_model("throughput_xgb_pred",X, y_throughput)
    bayesPred = BayesianModel.runBayesian("throughput_bayesPred","tgp",0.1,file_path)
    graph("tgp.png",y_throughput, "y_tgp", linear_pred, poly_pred, rbf_pred, rfr_pred, xgb_pred,bayesPred)
    # graph("tgp.png",y_throughput, "y_tgp",xgb_pred)

