import numpy as np 
import pandas as pd 
import eli5
import optuna
import joblib
import pickle
import gzip
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from pandas_profiling import ProfileReport

# Run optimization tool over the data 
def cost_ml_optimizer(trial, expected_values, vars_cost, vars_min_max, 
              ys_model, ts_model, el_model,
              lower_multiple_bnd=0.95, upper_multiple_bnd=1.05, penalty_factor=1000):
    """
    Using predictions from ML models optimizing the cost function 
    :param trial: object corresponds to a single execution of the objective function and is internally instantiated upon each invocation of the function.
    :param expected_values: the expected values of the mechanical properties of the steel i.e. yield_strength, tensile_strength and elongation.
    :param vars_cost: the cost per unit for each of the controllable variables i.e c, v, nb and mn.
    :param vars_min_max: the max and min values of controllable and non-controllable elements.
    :param ys_model: the yield_strength trained model.
    :param ts_model: the tensile_strength trained model.
    :param el_model: the elongation trained model.
    :param lower_multiple_bnd: lower bound constraint factor for the expected values. default is 0.95.
    :param upper_multiple_bnd: upper bound constraint factor for the expected values. default is 1.05.
    :return: best value if found after several trails. 
    """
    # setup the expected true values 
    true_ys = expected_values['yield_strength']
    true_ts = expected_values['tensile_strength']
    true_el = expected_values['elongation']
    
    # variables 
    c  = trial.suggest_float('c', vars_min_max['c'][0], vars_min_max['c'][1])
    mn = trial.suggest_float('mn', vars_min_max['mn'][0], vars_min_max['mn'][1])
    v  = trial.suggest_float('v', vars_min_max['v'][0], vars_min_max['v'][1])
    nb = trial.suggest_float('nb', vars_min_max['nb'][0], vars_min_max['nb'][1])
    p  = trial.suggest_float('p', vars_min_max['p'][0], vars_min_max['p'][1])
    s  = trial.suggest_float('s', vars_min_max['s'][0], vars_min_max['s'][1])
    si = trial.suggest_float('si', vars_min_max['si'][0], vars_min_max['si'][1])
    al = trial.suggest_float('al', vars_min_max['al'][0], vars_min_max['al'][1])
    cu = trial.suggest_float('cu', vars_min_max['cu'][0], vars_min_max['cu'][1])
    cr = trial.suggest_float('cr', vars_min_max['cr'][0], vars_min_max['cr'][1])
    ni = trial.suggest_float('ni', vars_min_max['ni'][0], vars_min_max['ni'][1])
    mo = trial.suggest_float('mo', vars_min_max['mo'][0], vars_min_max['mo'][1])
    ca = trial.suggest_float('ca', vars_min_max['ca'][0], vars_min_max['ca'][1])
    ti = trial.suggest_float('ti', vars_min_max['ti'][0], vars_min_max['ti'][1])
    sn = trial.suggest_float('sn', vars_min_max['sn'][0], vars_min_max['sn'][1])
    b  = trial.suggest_float('b', vars_min_max['b'][0], vars_min_max['b'][1])
    n  = trial.suggest_float('n', vars_min_max['n'][0], vars_min_max['n'][1])
    o  = trial.suggest_float('o', vars_min_max['o'][0], vars_min_max['o'][1])

    # define cost expression
    cost = vars_cost['c']*c + vars_cost['v']*v + vars_cost['nb']*nb + vars_cost['mn']*mn
    
    # setup data for prediction 
    trail_features = pd.DataFrame({'c':c, 'mn':mn, 'v':v, 'nb':nb, 'p':p, 's':s, 'si':si,
                                   'al':al, 'cu':cu, 'cr':cr, 'ni':ni, 'mo':mo, 'ca':ca,
                                   'ti':ti, 'sn':sn, 'b':b, 'n':n, 'o':o}, index=[0])
    
    # scoring of a model and evaluation if it meets constraint condition
    pred_ys = ys_model.predict(trail_features)
    pred_ts = ts_model.predict(trail_features)
    pred_el = el_model.predict(trail_features)
    
    if (pred_ys >= lower_multiple_bnd*true_ys and pred_ys <= upper_multiple_bnd*true_ys) \
    and (pred_ts >= lower_multiple_bnd*true_ts and pred_ts <= upper_multiple_bnd*true_ts) \
    and (pred_el >= lower_multiple_bnd*true_el and pred_el <= upper_multiple_bnd*true_el):
        return cost
    else:
        return cost*penalty_factor

# Compute the cost 
def get_cost(elements_cost, vars_cost):
    """
    Compute the cost for production of steel given the elements, quantities and cost per unit.
    :param elements_cost: a pandas.series object indexed with elements having matching quantities used for the steel production. 
                          also, it accepts dictionary whereby, the key is element.
    :param vars_cost: a pandas.DataFrame object havine the major elements and corresponding unit costs.
    """
    cost = 0
    
    if isinstance(elements_cost, pd.Series):
        params = elements_cost.to_dict()
    elif isinstance(elements_cost, dict):
        pass
    
    for var in elements_cost.keys():
        qty = elements_cost[var]
        if var in vars_cost.columns:
            cost += qty * vars_cost[var]
    return cost

# Score test cases to check if they meet lower & uppper bounded conditions
def get_ml_model_constraint_data(model, features, labels, lower_const=0.95, upper_const=1.05):
    
    

    """
    Score a given data based on a certain bounded predictive interval
    :param model: the trained model
    :param features: the features of the unseen or test data.
    :param labels: the with held labels of the unseen or test data.
    :param lower_const: lower bound constraint factor w.r.t the labels.
    :param upper_const: upper bound constraint factor w.r.t the labels.
    :return: a dataframe with corresponding scores which can be used to evaluate the predictive interval correctness of the model
    """
    preds = model.predict(features)
    preds_lower_bound = list(lower_const * labels)
    preds_upper_bound = list(upper_const * labels)
    test_labels = list(labels)
    c_ml = [1 if (preds[i] >= preds_lower_bound[i] and preds[i] <= preds_upper_bound[i]) else 0 for i in range(0, len(test_labels))]
    d = {'true_values':test_labels, 
         'pred_values':preds, 
         'pred_lower_bound':preds_lower_bound, 
         'pred_upper_bound':preds_upper_bound,
         'constraint_score': c_ml
        }
    return pd.DataFrame(d)

# Get cross validation scores
def get_ml_model_cross_validation_scores(model, features, label, n, metric='r2'):
    """
    Evaluate a score by cross-validation, a number of times.
    :param model: the model object
    :param features: the features used for developing the model
    :param label: the corresponding target
    :param n: the number of times the cross validation needs to be done
    :param metric: the kpi to be used for evaluation and reporting
    :return: an array of scores based on the chosen kpi. by default the r2 will be returned.
    """
    all_scores = cross_val_score(model, features, label, cv=n, scoring=metric)
    print('cross-validation mean: ', np.mean(all_scores))
    print('standard deviation: ', np.std(all_scores))
    print('\n')
    return all_scores 

# Get the performance of the model on validation data 
def get_ml_model_performance(model, features, labels):
    """
    Predict and evaluate a given trained ML model on an unseen data.
    :param features: the features of the unseen or test data.
    :param labels: the with held labels of the unseen or test data.
    :return: the r2 score.
    """
    from sklearn import metrics 
    predictions = model.predict(features)
    errors = abs(predictions - labels)
    mape = 100 * np.mean(errors / labels)
    accuracy = 100 - mape
    r2 = metrics.r2_score(labels, predictions) * 100
    print('model performance')
    #print('average error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('acc = {:0.2f}%'.format(r2))
    return r2

# Evaluate the scored data 
def get_ml_model_constraint_performance(ml_constraint_data):
    """
    Evaluate a scored model based on some constrain bounded conditions
    :param ml_constraint_data: the scored data from get_ml_model_constraint_data function.
    :return: a dataframe showing the accuracy based on constraint accuracy
    """
    absolute = ml_constraint_data.constraint_score.value_counts()
    relative = ml_constraint_data.constraint_score.value_counts(normalize=True) * 100
    results = pd.concat([absolute, relative], axis=1)
    results.columns = ['absolute', 'relative']
    results.rename(index={1:'constraint success', 0:'constraint failed'}, inplace=True)
    return results

# Understanding the distributions of the Error
def get_ml_model_boxplot(ml_model, experiments, actuals):
    """
    Get the boxplot of the relative errors
    :param ml_model: the name of the algorithm to be boxplotted.
    :param experiments: the data to be plotted.
    :param actuals: the true target values.
    :return: returns boxplot
    """
    prediction = (experiments["Predictions"]
                 [experiments["Algorithm"].index(ml_model)])
    plt.title(ml_model)
    plt.boxplot((prediction - actuals) / (1 + abs(actuals)))
    plt.show()

def save_zipped_joblib(obj, filename):
    """
    Compress and save the pickle file so that it can be uploaded into git
    :param obj: object to be saved.
    :param filename: name of the object to be saved.
    :return: nothing 
    """
    with gzip.open(filename, 'wb') as f:
        joblib.dump(obj, f)

def load_zipped_joblib(filename):
    """
    Read compressed pickle object and uncompress it
    :param filename: name of the object to be read.
    :return: returns pickle object
    """
    with gzip.open(filename, 'rb') as f:
        loaded_object = joblib.load(f)
        return loaded_object
