#!/usr/bin/env python
# coding: utf-8

# # Model for shark and ray meat trade
# 
# This code implements a probabalistic model to estimate the proportions of species traded among major shark fishing and trading nations. A previous model uses expert opinion as prior information to estimate the total latent landings in any given country and this model allocates these landings to potential trading partners based on trade friction and importer preferences. Model developmenet has been primarily from Aaron MacNeil, Beth Babcock, Chris Mull, and Alex Andorra.
# 

# In[1]:


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
import xarray_einstats
import rdata as rd
import mcbackend
import clickhouse_driver
import networkx as nx
import itertools


# In[2]:


# Set figure style.
az.style.use("arviz-darkgrid")
# point to data and figure directories
bd = os.getcwd() + "/../Data/"
bf = os.getcwd() + "/../Figures/"

pd.set_option('display.max_rows', 500)


# In[3]:


# Helper functions
def indexall(L):
    poo = []
    for p in L:
        if not p in poo:
            poo.append(p)
    Ix = np.array([poo.index(p) for p in L])
    return poo, Ix


# Helper functions
match = lambda a, b: np.array([b.index(x) if x in b else None for x in a])

def zscore(x):
    return (x-np.mean(x))/np.std(x)

def unique(series: pd.Series):
    "Helper function to sort and isolate unique values of a Pandas Series"
    return series.sort_values().unique()


# Script to generate interpolated PyMC distribution objects
def from_posterior(param, samples):
    smin, smax = samples.min().item(), samples.max().item()
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = sp.stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return pm.Interpolated(param, x, y)



# # Load data

# In[4]:


# Load data
exec(open("Joint_Trade_Landings_Data_AUGM.py").read())


# In[5]:


# Initalize backend
#ch_client = clickhouse_driver.Client(host="xxx.xxx.xxx.xxx", password='your-pwd', database='gsmtdb')
# Backend object
#ch_backend = mcbackend.ClickHouseBackend(ch_client)


# In[6]:


# List backend runs available
#rxid = ch_backend.get_runs()
#rxid


# In[7]:


# Fetch the run from the database (downloads just metadata from most recent run)
#model_run = ch_backend.get_run('XXXX')


# In[8]:


# Import MultiTrace objects from server
#IDATA = model_run.to_inferencedata(var_names=['latent_logOdds_landings_'])


# In[9]:


# Import landings log-odds posteriors
IDATA = az.from_netcdf("idata-landings-model_AUGM.nc")

latent_logOdds_landings_mu = IDATA.posterior['latent_logOdds_landings_'].mean(('chain','draw')).to_numpy()
latent_logOdds_landings_sd = IDATA.posterior['latent_logOdds_landings_'].std(('chain','draw')).to_numpy()

# Remove IDATA object
IDATA = 0


# In[11]:


# Mask for dyads
dyad_mask = (trade_mask.max(1)!=-999)*1
# Mask for importer species prefs
ispp_mask = (trade_mask.max(0)!=-999)*1
# Log-seafood trade and mask
total_seafood_trade[total_seafood_trade<=1] = 0
log_total_seafood_trade = np.log1p(total_seafood_trade.to_numpy())
lst_mask = (log_total_seafood_trade>0)*1


# # Joint landings trade model - sharks and rays

# In[12]:


with pm.Model(coords=COORDS) as trade_model_x:
    # --------------------------------------------------------
    #                     Data containers
    # --------------------------------------------------------
    
    shark_exporter_id = pm.Data(
        "shark_exporter_id", shark_exporter_idx, dims="shark_obs_idx"
    )
    shark_importer_id = pm.Data(
        "shark_importer_id", shark_importer_idx, dims="shark_obs_idx"
    )
    ray_exporter_id = pm.Data(
        "ray_exporter_id", ray_exporter_idx, dims="ray_obs_idx"
    )
    ray_importer_id = pm.Data(
        "ray_importer_id", ray_importer_idx, dims="ray_obs_idx"
    )
    unreliability_data = pm.Data(
        "unreliability_data", unreliability_score["exporter"], dims="exporter"
    )
    log_seafood_trade = pm.Data(
        "log_seafood_trade",
        np.log(total_seafood_trade+0.0001),
        dims=("exporter", "importer"),
    )
    z_seafood_trade = pm.Data(
        "z_seafood_trade",
        zscore(np.log(total_seafood_trade.to_numpy()+0.0001)),
        dims=("exporter", "importer"),
    )
    log_shark_trade_data = pm.Data(
        "log_shark_trade_data",
        np.log(tdata_cut_sharks["estimated_live_weight"]),
        dims="shark_obs_idx",
    )
    shark_trade_data = pm.Data(
        "shark_trade_data",
        tdata_cut_sharks["estimated_live_weight"],
        dims="shark_obs_idx",
    )
    log_ray_trade_data = pm.Data(
        "log_ray_trade_data",
        np.log(tdata_cut_rays["estimated_live_weight"]),
        dims="ray_obs_idx",
    )
    ray_trade_data = pm.Data(
        "ray_trade_data",
        tdata_cut_rays["estimated_live_weight"],
        dims="ray_obs_idx",
    )
    TradeWeights = pm.Data(
        "TradeWeights",
        tradeweights.to_numpy(),
        dims=("exporter", "species", "importer"),
    )

    # --------------------------------------------------------
    #                   Landings model
    # --------------------------------------------------------

    #"""
    # Set index cutoff - removes impossible -999 species from likelihood
    isppx = SppPRIORadj > -5


    # National average log odds for latent landings by species
    latent_logOdds_landings_ = pm.Normal("latent_logOdds_landings_", latent_logOdds_landings_mu, latent_logOdds_landings_sd+0.1)

    latent_logOdds_landings = pm.Deterministic(
        "latent_logOdds_landings",
        pyt.set_subtensor((pyt.ones(SppPRIOR.shape)*-9)[isppx], latent_logOdds_landings_),
        dims=("exporter", "species")
    )

    # Species proportions of total landings
    SppProps = pm.Deterministic(
        "SppProps",
        pm.math.softmax(latent_logOdds_landings,axis=1),
        dims=("exporter", "species")
    )
    
    # National average landings
    Latent_landings = pm.Deterministic(
        "Latent_landings",
        SppProps*TotalCatch[:,None],
        dims=("exporter", "species")
    )
    
    # ----------------------------------------------------------------------------------------------------------------
    #           TRADE MODEL
    # ----------------------------------------------------------------------------------------------------------------

    intercept = pm.Normal("intercept", 0, 3)
    
    #importer_species_effect = pm.Normal("importer_species_effect", 0, 3, dims=("species", "importer"))
    importer_species_effect = pm.ZeroSumNormal("importer_species_effect", 3, dims=("species", "importer"), n_zerosum_axes=2)
    
    #dyad_effect = pm.Normal("dyad_effect", 0, 3, dims=("exporter", "importer"))
    dyad_effect = pm.ZeroSumNormal("dyad_effect", 3, dims=("exporter", "importer"), n_zerosum_axes=2)
    
    # trade affinity between countries
    trade_effect = pm.Normal("trade_effect", 0, 3)
    
    # exporters' propensity to sell species
    # doesn't have to sum to 1 across species
    # has to sum to 1 across importers
    export_proportion = pm.Deterministic(
        "export_proportion", 
        pm.math.softmax(
            (intercept
            + importer_species_effect[None, :, :]
            + dyad_effect[:, None, :]
            + (trade_effect * z_seafood_trade)[:, None, :]
            + trade_mask
            + TradeWeights
            )
            , axis=2
        ),
        dims=("exporter", "species", "importer")
    )
    
    # Amount exported
    amount_exported = pm.Deterministic(
        "amount_exported",
        ((export_proportion)*(Latent_landings[:, :, None])),
        dims=("exporter", "species", "importer"),
    )

    # --------------------------------------------------------
    #                    Sum across species
    # --------------------------------------------------------

    expected_log_shark_trade = pm.Deterministic(
        "expected_log_shark_trade",
        pyt.log((amount_exported*shark_mask[None,:,None]).sum(1)+0.0001),
        dims=("exporter", "importer"),
    )
    expected_log_ray_trade = pm.Deterministic(
        "expected_log_ray_trade",
        pyt.log((amount_exported*ray_mask[None,:,None]).sum(1)+0.0001),
        dims=("exporter", "importer"),
    )
    
    # --------------------------------------------------------
    #       Modeling exporters' reporting unreliability
    # --------------------------------------------------------

    lsd_intercept_sd = pm.Exponential("lsd_intercept_sd", 1)
    lsd_intercept = pm.Normal("lsd_intercept", sigma=lsd_intercept_sd)

    lsd_unreliability_sd = pm.Exponential("lsd_unreliability_sd", 1)
    lsd_unreliability_effect = pm.Normal(
        #"lsd_unreliability_effect", sigma=lsd_unreliability_sd, dims="exporter"
        "lsd_unreliability_effect", sigma=lsd_unreliability_sd, dims="exporter"
    )

    reliability = pyt.exp(
        lsd_intercept + lsd_unreliability_effect * unreliability_data
    )
    #"""
    # --------------------------------------------------------
    #                    Likelihood of trade
    # --------------------------------------------------------
    #"""
    pm.Normal(
        "log_shark_trade",
        mu=expected_log_shark_trade[shark_exporter_id, shark_importer_id],
        sigma=reliability[shark_exporter_id],
        observed=log_shark_trade_data,
        dims="shark_obs_idx",
    )
    
    pm.Normal(
        "log_ray_trade",
        mu=expected_log_ray_trade[ray_exporter_id, ray_importer_id],
        sigma=reliability[ray_exporter_id],
        observed=log_ray_trade_data,
        dims="ray_obs_idx",
    )
    #"""


# In[13]:


with trade_model_x:
    #trade_model_x = pm.sample(draws=500, tune=1000, trace=ch_backend, idata_kwargs=dict(log_likelihood=False))
    idata_trade_x = pm.sample(draws=100, tune=100, idata_kwargs=dict(log_likelihood=False))
print('Done sampling')


# In[14]:


# Sample from prior and posterior predictive distributions
#"""
with trade_model_x:
    try:
        idata_trade_x.extend(pm.sample_prior_predictive())
        idata_trade_x.extend(pm.sample_posterior_predictive(idata_trade_x))
        print('Done predictives')
    except:
        print('Failed predictives')
#"""


# In[15]:


# Export results
idata_trade_x.to_netcdf("idata-trade-model_AUGM.nc")
print("Saved trace!")

