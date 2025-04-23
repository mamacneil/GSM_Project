#!/usr/bin/env python
# coding: utf-8
# Process mcbackend model rund for local use

# Pulls traces from clickhouse backend, using mcbackend, into netcdf files stored locally
# In[2]:


import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pyt
import seaborn as sns
import pdb
from matplotlib.gridspec import GridSpec
import xarray as xr
import rdata as rd
import mcbackend as mb
import clickhouse_driver


# In[3]:


# Set figure style.
az.style.use("arviz-darkgrid")
bd = os.getcwd() + "/../Data/"
bf = os.getcwd() + "/../Figures/"


# # Import inference object

# In[4]:


# Initalize backend
#ch_client = clickhouse_driver.Client(host="129.173.118.118", password='buKcek-qetsyj-pynci7', database='gsmtdb')
ch_client = clickhouse_driver.Client(host="129.173.118.118", password='buKcek-qetsyj-pynci7', database='gsmtdb', send_receive_timeout = 7200, settings={'max_execution_time': 7200})
# Backend object
ch_backend = mb.ClickHouseBackend(ch_client)
# List backend runs available
#ch_backend.get_runs()


# In[5]:


# Fetch the run from the database (downloads just metadata)
model_run = ch_backend.get_run('C341T')


# In[ ]:


# Import MultiTrace objects from server
idata_landings_a = model_run.to_inferencedata(var_names=['Latent_landings','CountrySPP_landings','CountrySPP_Taxonlandings','latent_logOdds_landings'])


# In[ ]:
print('Run imported!')

# Export results
idata_landings_a.to_netcdf("idata-landings-model_AUGM_Perth_server.nc")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




