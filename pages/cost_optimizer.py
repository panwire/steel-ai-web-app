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
from halo import HaloNotebook as Halo
from pages import utils
from datetime import date


#@st.cache
def app():
    st.markdown("## Cost Optimizer")

    # Upload the dataset and save as csv
    st.markdown("### Developing An Optimal Cost Strategy For Steel Production") 
    st.markdown("""
                * Use the menu below to select and provide the necessary inputs (controllable variables) for the cost optimizer artificial intelligence system.
                * The results will appear below after a successful run has been executed.
    """)
    st.write("\n")
    
    today = date.today()
    form = st.form(key="annotation")
    
    df = pd.read_excel("./data/hsm_for_app.xlsx")
    df.columns = [name.lower().replace(' ', '_') for name in df.columns]
    df = df.set_index('heat_no')
    uniq_heat_nos = list(df.index.unique())
    
    controllables = ['c', 'mn', 'v', 'nb']
    non_controllables = ['p', 's', 'si', 'al', 'cu', 'cr', 'ni', 'mo', 'ca', 'ti', 'sn', 'b', 'n', 'o']
    input_cols = controllables + non_controllables
    target_cols = ['yield_strength', 'tensile_strength', 'elongation']

    
    with form:
        cols = st.columns((1, 1))
        heat_no = cols[0].selectbox('Heat Column ID', uniq_heat_nos, help = 'Please, select only one heat column at a time', index=1)
        n_exp = cols[1].selectbox("Number Of Experiment Runs:", [100, 1000, 2000, 3000, 4000, 5000], #list(range(0, 6000, 1000))[1:], 
                                  help = 'Please, select the number of times the trails to be executed by the optimizer. higer the number the greater the chance of finding an optimal solution but the search will take long.', index=2)
        cols = st.columns(4)
        c_per_unit = cols[0].text_input('Carbon Cost per Unit ($):', value="16", help='Please, enter the cost per unit for carbon in dollars.')
        mn_per_unit = cols[1].text_input('Manganese Cost per Unit ($):', value="104", help = 'Please, enter the cost per unit for manganese in dollars.')
        nb_per_unit = cols[2].text_input('Niobium Cost per Unit ($):', value="6050", help = 'Please, enter the cost per unit for niobium in dollars.')
        v_per_unit = cols[3].text_input('Vanadium Cost per Unit ($):', value="3250", help = 'Please, enter the cost per unit for vanadium in dollars.')
        cols = st.columns(2)
        lower_bound = cols[0].selectbox('Lower Bound For Predictive Interval (%):', [-5, -10], help='Please, enter the lower bound for predictive interval for the heat column target values in percentage.')
        upper_bound = cols[1].selectbox('Upper Bound For Predictive Interval (%):', [5, 10], help='Please, enter the upper bound for predictive interval for the heat column target values in percentage.')


        submitted = st.form_submit_button(label="Submit")
        
    if submitted:
        # Preparing Optimizer 
        st.info("Thanks! You have successful made an execution. Please, patiently wait for the results...")
        n_exp = int(n_exp)
        l_bnd = 1 + (int(lower_bound) / 100)
        u_bnd = 1 + (int(upper_bound) / 100)
        vars_cost = pd.DataFrame({'c': int(c_per_unit), 'v': int(v_per_unit), 'nb': int(nb_per_unit), 'mn': int(mn_per_unit)}, index=[0])
        vars_min_max = pd.read_excel("./data/vars_min_max_for_app.xlsx")
        vars_min_max.set_index('index')
        expected_values = df.loc[heat_no, target_cols]
        
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
        actual_input_values = df.loc[heat_no, input_cols]
        actual_output_values = df.loc[heat_no, target_cols]
        
        # Prepare the essential results for display
        actual_cost = round(float(utils.get_cost(elements_cost=actual_input_values, vars_cost=vars_cost)), 2)
        optimized_cost = round(float(utils.get_cost(elements_cost=best_params, vars_cost=vars_cost)), 2)
        saving = actual_cost - optimized_cost
        lift = round((actual_cost / optimized_cost) - 1, 2)
        roi  = round((saving) / actual_cost, 4) * 100
        
        actual_df = actual_input_values[controllables]
        report1 = pd.DataFrame({'report_date': [today, today],
                               'heat_no':[heat_no, heat_no],
                               'steel_production_strategy': ['actual solution', 'ai optimized solution'],
                               'trails': [None, int(n_exp)],
                               'processing_time (mins)': [None, total_mins],
                               'c': [actual_df['c'], best_params['c']], 
                               'v': [actual_df['v'], best_params['v']],
                               'mn': [actual_df['mn'], best_params['mn']],
                               'nb': [actual_df['nb'], best_params['nb']],
                               'cost ($)': [round(actual_cost,4), round(optimized_cost,4)],
                               })
        
        optimized_ys = utils.get_ml_model_constraint_data(ys_model, best_params_x, pd.Series(actual_output_values[0]), lower_const=l_bnd, upper_const=u_bnd)
        optimized_ys['target'] = 'yield_strength'
        optimized_ts = utils.get_ml_model_constraint_data(ts_model, best_params_x, pd.Series(actual_output_values[1]), lower_const=l_bnd, upper_const=u_bnd)
        optimized_ts['target'] = 'tensile_strength'
        optimized_el = utils.get_ml_model_constraint_data(el_model, best_params_x, pd.Series(actual_output_values[2]), lower_const=l_bnd, upper_const=u_bnd)
        optimized_el['target'] = 'elongation'
        report2 = pd.concat([optimized_ys, optimized_ts, optimized_el], axis=0)
        report2.reset_index(inplace=True, drop=True)
        report2 = report2[['target', 'true_values', 'pred_values', 
                           'pred_lower_bound', 'pred_upper_bound', 'constraint_score']]

        
        fig = go.Figure(data=[go.Bar(name='Lower Bound', x=report2['target'], y=report2['pred_lower_bound']),
                              go.Bar(name='Predicted Value', x=report2['target'], y=report2['pred_values']),
                              go.Bar(name='True Value', x=report2['target'], y=report2['true_values']),
                              go.Bar(name='Upper Bound', x=report2['target'], y=report2['pred_upper_bound'])
                              ])
        fig.layout.update(barmode='group', title_text=f'Predicting New Design For Heat Coloumn ID {heat_no} At Reduced Cost', title_x=0.5)
        
        # Display results
        #st.info("Thanks for your patience! Here are your results.")
        m1, m2, m3, m4 = st.columns((1, 1, 1, 1))
        m1.write('')
        m2.metric(label="Cost Saving ($)", value=round(saving, 4), delta = str(round(roi,4))+ '% ROI.', delta_color = "normal")
        m3.metric(label="Cost Lift", value=round(lift, 4), delta = str(round((lift)*100, 4)) + '% Improvement.', delta_color = "normal")
        m4.write('')
        
        expander = st.expander("Insight Reports")
        with expander:
            st.table(report1)
            st.plotly_chart(fig, use_container_width=True)
            st.table(report2)
            
        st.balloons()

    
