#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This app accomplishes serve two objectives:- 
    1) find the minimize the cost of steel production
    2) find the relatively best amount of materials needed for steel production
"""

###### Packages #########
import optuna
import joblib
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import pandasql.sqldf as exec_sql
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

###### Class, Methods & Functions #########
from multipage import MultiPage
from pages import cost_optimizer, steel_builder

###### Start Page #########
# Create an instance of the app 
app = MultiPage()
apptitle = 'Steel Production App'
st.set_page_config(page_title=apptitle,  layout='wide',  page_icon=':rocket:')

# Title the app
#st.title('Gravitational Wave Quickview')

###### Seperate Pages #########
app.add_page("Cost Optimizer", cost_optimizer.app)
app.add_page("Steel Builder", steel_builder.app)


# The main app
app.run()

# Disable foot note 
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
