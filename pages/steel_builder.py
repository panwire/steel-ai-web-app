import eli5
import optuna
import joblib
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import datetime
from matplotlib.pyplot import cm
from pages import utils
from datetime import date


#@st.cache
def app():
    st.markdown("## Artificial Intelligence (AI) Steel Builder")

    # Upload the dataset and save as csv
    st.markdown("### This AI Application Helps Find The Optimal Amount Of Materials Needed For Developing Steel Of A Desirable Quality") 
    st.markdown("""
                * Use the menu below to provide the desirable mechanical properties of the steel.
                * The results will appear below after a successful run has been executed.
    """)
    st.write("\n")
    
    today = date.today()
    form = st.form(key="annotation")
    
    vars_min_max = pd.read_excel("./data/vars_min_max_for_app.xlsx")
    vars_min_max.set_index('index')
    
    controllables = ['c', 'mn', 'v', 'nb']
    non_controllables = ['p', 's', 'si', 'al', 'cu', 'cr', 'ni', 'mo', 'ca', 'ti', 'sn', 'b', 'n', 'o']
    input_cols = controllables + non_controllables
    target_cols = ['yield_strength', 'tensile_strength', 'elongation']
    
    with form:
        cols = st.columns((3))
        ys = cols[0].text_input('Yield Strength (N/m²)', value="28.1", help='Please, enter the desirable value for yield strength you might like to attain. The yield strength of steel represents the stress beyond which its deformation is plastic. Ideally, consider keeping values between 0 to 100.')
        ts = cols[1].text_input('Tensile Strength (N/m²)', value="40.3", help='Please, enter the desirable value for tensile strength you might like to attain. The tensile strength is the maximum tensile load the steel can withstand prior to fracture. Ideally, consider keeping values between 0 to 100.')
        el = cols[2].text_input('Elongation (%)', value="34", help='Please, enter the desirable value for elongation you might like to attain. Elongation is the percentage of stretch from the original length of the steel to the point of failure, showing how ductile the steel is. Ideally, consider keeping values between 0 to 100.')
        cols = st.columns(4)
        c_per_unit = cols[0].text_input('Carbon Cost per Unit ($):', value="16", help='Please, enter the cost per unit for carbon in dollars.')
        mn_per_unit = cols[1].text_input('Manganese Cost per Unit ($):', value="104", help = 'Please, enter the cost per unit for manganese in dollars.')
        nb_per_unit = cols[2].text_input('Niobium Cost per Unit ($):', value="6050", help = 'Please, enter the cost per unit for niobium in dollars.')
        v_per_unit = cols[3].text_input('Vanadium Cost per Unit ($):', value="3520", help = 'Please, enter the cost per unit for vanadium in dollars.')
        cols = st.columns(3)
        lower_bound = cols[0].selectbox('Lower Bound For Predictive Interval (%):', [-1, -2, -3, -4, -5], help='Please, enter the lower bound in percentage. This will be applied to all the target values setting above.')
        upper_bound = cols[1].selectbox('Upper Bound For Predictive Interval (%):', [1, 2, 3, 4, 5], help='Please, enter the upper bound in percentage. This will be applied to all the target values setting above.')
        n_exp = cols[2].selectbox("Number Of Experiment Runs:", [1000, 2000, 3000, 4000, 5000],
                                  help = 'Please, select the number of times the trails to be executed by the optimizer. higer the number the greater the chance of finding an optimal solution but the search will take long.', index=2)
        
        submitted = st.form_submit_button(label="Submit")
        #st.markdown("""The above are the automated column types detected by the application in the data. In case you wish to change the column types, head over to the **Column Change** section. """)
        
    if submitted:
        d = {'yield_strength': float(ys), 'tensile_strength': float(ts), 'elongation': float(el)}
        expected_values = pd.Series(data=d, index=['yield_strength', 'tensile_strength', 'elongation'])
        vars_cost = pd.DataFrame({'c': int(c_per_unit), 'v': int(v_per_unit), 'nb': int(nb_per_unit), 'mn': int(mn_per_unit)}, index=[0])
        n_exp = int(n_exp)
        l_bnd = 1 + (int(lower_bound) / 100)
        u_bnd = 1 + (int(upper_bound) / 100)
        
        # Load predictive models
        ys_model = utils.load_zipped_joblib(f'./models/{target_cols[0]}.joblib')
        ts_model = utils.load_zipped_joblib(f'./models/{target_cols[1]}.joblib')
        el_model = utils.load_zipped_joblib(f'./models/{target_cols[2]}.joblib')
        
        # Run Optimizer
        study = optuna.create_study(direction='minimize')
        start_time = datetime.datetime.now()
        study.optimize(lambda trial: utils.cost_ml_optimizer(trial, 
                                                             expected_values=expected_values, 
                                                             vars_cost=vars_cost, 
                                                             vars_min_max=vars_min_max, 
                                                             ys_model=ys_model, ts_model=ts_model, el_model=el_model,
                                                             lower_multiple_bnd=l_bnd, upper_multiple_bnd=u_bnd), n_trials=n_exp)
        end_time = datetime.datetime.now()
        time_diff = end_time - start_time
        total_mins = round(time_diff.seconds  / 60)
        
        # Compute the results 
        best_params = study.best_params
        best_params_x = pd.DataFrame(best_params, index=[0])
        optimized_cost = round(float(utils.get_cost(elements_cost=best_params, vars_cost=vars_cost)), 2)

        # Reporting 
        report1 = pd.DataFrame({'report_date': [today],
                               'steel_production_strategy': ['ai steel builder'],
                               'trails': [int(n_exp)],
                               'processing_time (mins)': [total_mins],
                               'c': [best_params['c']], 
                               'v': [best_params['v']],
                               'mn': [best_params['mn']],
                               'nb': [best_params['nb']],
                               'p': [best_params['p']],
                               's': [best_params['s']],
                               'si': [best_params['si']],
                               'al': [best_params['al']],
                               'cu': [best_params['cu']],
                               'cr': [best_params['cr']],
                               'ni': [best_params['ni']],
                               'mo': [best_params['mo']],
                               'ca': [best_params['ca']],
                               'ti': [best_params['ti']],
                               'sn': [best_params['sn']],
                               'b': [best_params['b']],
                               'n': [best_params['n']],
                               'o': [best_params['o']],
                               'cost ($)': [round(optimized_cost,4)],
                               })
        
        optimized_ys = utils.get_ml_model_constraint_data(ys_model, best_params_x, pd.Series(expected_values[0]), lower_const=l_bnd, upper_const=u_bnd)
        optimized_ys['target'] = 'yield_strength'
        optimized_ts = utils.get_ml_model_constraint_data(ts_model, best_params_x, pd.Series(expected_values[1]), lower_const=l_bnd, upper_const=u_bnd)
        optimized_ts['target'] = 'tensile_strength'
        optimized_el = utils.get_ml_model_constraint_data(el_model, best_params_x, pd.Series(expected_values[2]), lower_const=l_bnd, upper_const=u_bnd)
        optimized_el['target'] = 'elongation'
        report2 = pd.concat([optimized_ys, optimized_ts, optimized_el], axis=0)
        report2.reset_index(inplace=True, drop=True)
        report2 = report2[['target', 'true_values', 'pred_values', 
                           'pred_lower_bound', 'pred_upper_bound', 'constraint_score']]

        constraints = report2['constraint_score'].unique()
        if len(constraints) == 1:
            if int(constraints) == 0:
                mgs = """
                      The solution found did not meet constraint conditions. Please, try to do the followings:-
                            <br>1) Increase the number of trails.<\br>
                            <br>2) Increase the boundary predictive interval.<\br>
                            <br>3) If the above two suggestions do not work, change the target mechanical values.<\br>
                      """
            else:
                mgs = """
                        Congratulations, You found a solution!
                      """
        else:
            mgs = """
                    Unfortunately, some mechanical properties were not met. Please, consider adjusting some of the input parameters:-
                            <br>1) Increase the number of trails.<\br>
                            <br>2) Increase the boundary predictive interval.<\br>
                            <br>3) If the above two suggestions do not work, change the target mechanical values.<\br>
                  """
        
        fig = go.Figure(data=[go.Bar(name='Lower Bound', x=report2['target'], y=report2['pred_lower_bound']),
                              go.Bar(name='Predicted Value', x=report2['target'], y=report2['pred_values']),
                              go.Bar(name='True Value', x=report2['target'], y=report2['true_values']),
                              go.Bar(name='Upper Bound', x=report2['target'], y=report2['pred_upper_bound'])
                              ])
        fig.layout.update(barmode='group', title_text=f'Steel AI Builder Outcome', title_x=0.5)
        
        # Display results
        st.markdown(mgs)
        m1, m2, m3 = st.columns((1, 1, 1))
        m1.write('')
        m2.metric(label="Material Cost ($)", value=round(optimized_cost, 4))
        m3.write('')
        
        expander = st.expander("Insight Reports")
        with expander:
            st.dataframe(report1)
            st.plotly_chart(fig, use_container_width=True)
            st.table(report2)
            
        st.balloons()

    
