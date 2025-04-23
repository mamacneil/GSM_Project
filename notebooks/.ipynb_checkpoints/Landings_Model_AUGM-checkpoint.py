#!/usr/bin/env python
# coding: utf-8

# # Model for shark and ray meat landings
# 
# This code implements a probabalistic model to estimate the proportions of species landed among major shark fishing and trading nations. The model uses expert opinion as prior information to estimate the total latent landings in any given country. Model developmenet has been primarily from Aaron MacNeil, Beth Babcock, Chris Mull, and Alex Andorra.
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


# In[2]:


# Set figure style.
az.style.use("arviz-darkgrid")
# point to data and figure directories
bd = os.getcwd() + "/../Data/"
bf = os.getcwd() + "/../Figures/"


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


# # Load data

# In[4]:


# Load data
exec(open("Joint_Trade_Landings_Data_AUGM.py").read())


# In[5]:


# Initalize backend
#ch_client = clickhouse_driver.Client(host="129.173.118.118", password='buKcek-qetsyj-pynci7', database='gsmtdb')
#ch_client = clickhouse_driver.Client(host="129.173.118.118", password='buKcek-qetsyj-pynci7', database='gsmtdb', send_receive_timeout = 7200, settings={'max_execution_time': 7200})
# Backend object
#ch_backend = mcbackend.ClickHouseBackend(ch_client)


# In[6]:


# Set index cutoff - removes impossible -999 species from likelihood
isppx = SppPRIOR > -5
# Add +1 to possible species
#ObsLandings_obs[isppx[obs_exporter_idx,:]] += 1
# Observed totals for multinomial
#ObsLandings_sums = ObsLandings.astype(int).sum(axis=1).values
ObsLandings_sums = ObsLandings.sum(axis=1).values
# Observed values for multinomial
#ObsLandings_obs = ObsLandings.astype(int).values
ObsLandings_obs = ObsLandings.values

# Index for non-zeros
zindx = ObsLandings_obs>0
# Log data
logObsLandings_obs = np.log(ObsLandings_obs[zindx])
logObsLandings_obs2 = np.log(ObsLandings_obs+0.00001)


# In[7]:


# Set index cutoff - removes impossible -999 and non retained species from likelihood
isppx = SppPRIOR > -5
nspp = TaxonMASK_t3.shape[1]
# Grab number of species available per country
nspp_country = np.tile((1-(isppx==0)*1).sum(1),(nspp,1)).T

# Grab taxon dims and numbers of aggregations per exporter
ntax = TaxonMASK_t3.shape[2]
ntaxon_country = np.tile(np.array([(x.sum(0)>0).sum() for x in TaxonMASK_t3]),(ntax,nspp,1)).T


# # Latent landings model - sharks and rays

# In[8]:


with pm.Model(coords=COORDS) as landings_model_x:
    # --------------------------------------------------------
    #                     Data containers
    # --------------------------------------------------------
    
    shark_exporter_id = pm.ConstantData(
        "shark_exporter_id", shark_exporter_idx, dims="shark_obs_idx"
    )
    shark_importer_id = pm.ConstantData(
        "shark_importer_id", shark_importer_idx, dims="shark_obs_idx"
    )
    ray_exporter_id = pm.ConstantData(
        "ray_exporter_id", ray_exporter_idx, dims="ray_obs_idx"
    )
    ray_importer_id = pm.ConstantData(
        "ray_importer_id", ray_importer_idx, dims="ray_obs_idx"
    )
    unreliability_data = pm.ConstantData(
        "unreliability_data", unreliability_score["exporter"], dims="exporter"
    )

    # --------------------------------------------------------
    #                   Landing data model
    # --------------------------------------------------------
    
    # = = = = = = = = = = = = = = = = Latent Species  = = = = = = = = = = = = = = = = #

    # Scale exporter variance to number of species
    plo_intercept = pm.Normal('plo_intercept',np.log(3),2)
    plo_numspp = pm.Normal('plo_numspp')
    Spp_magnitude = pm.Deterministic('Spp_PLO', pyt.exp(plo_intercept+plo_numspp*np.log(nspp_country+0.0001)) )
    
    #'''
    # National average log odds for latent landings by species
    latent_logOdds_landings_ = pm.Normal("latent_logOdds_landings_", SppPRIORadj[isppx], Spp_magnitude[isppx])
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
    #"""
    # = = = = = = = = = = = = = = = = OBSERVED DATA PROPORTIONS = = = = = = = = = = = = = = = = #

    # National average landings
    logLatent_landings_obs = pm.Deterministic(
        "logLatent_landings_obs",
        pyt.log(SppProps[obs_exporter_idx,:].eval()*ObsLandings_sums[:,None]+0.00001),
        dims=("obs_exporter", "species")
    )

    # Uncertainty
    ObsSD = pm.Exponential('ObsSD',1)

    # Likelihood
    pm.Normal(
        "logObserved_landings",
        logLatent_landings_obs[zindx],
        ObsSD,
        observed=logObsLandings_obs,
    )
    
    # = = = = = = = = = = = = = = = = SPECIES VS TAXON PROPORTIONS = = = = = = = = = = = = = = = = #

    # log-odds for latent landings identified to species -
    # CAN ADD COVARIATES HERE for TO SPP or AGGGREGATED E.G. GEAR, CITES, FISHERY, etc.
    PsppIdent_odds = pm.Normal("PsppIdent_odds", 0, 2, shape=SppPRIOR[isppx].shape)
    PsppIdent_ = pm.Deterministic(
        "PsppIdent_ ", pm.math.invlogit(PsppIdent_odds + NoTaxaSppWT[isppx]*10)
    )
    # Expand to full rank - make impossibles guaranteed to be spp ID'd
    PsppIdent = pm.Deterministic(
        "PsppIdent",
        pyt.set_subtensor(pyt.ones(SppPRIOR.shape)[isppx], PsppIdent_),
        dims=("exporter", "species"),
    )

    # = = = = = = = = = = = = = = = = SPECIES Model = = = = = = = = = = = = = = = = #

    # National average reported log landings by species
    CountrySPP_landings = pm.Deterministic(
        "CountrySPP_landings",
        PsppIdent*Latent_landings,
        dims=("exporter", "species"),
    )
    
    # National average reported log landings by species
    CountrySPP_log_landings = pm.Deterministic(
        "CountrySPP_log_landings",
        pyt.log(CountrySPP_landings+0.00001),
        dims=("exporter", "species")
    )
    # Allow for annual landings variation
    CountrySPP_log_yr = pm.ZeroSumNormal('CountrySPP_log_yr',dims=("exporter", "species", "year"),n_zerosum_axes=2)

    # Add on extra bit
    CountrySPP_log_landings_yr = pm.Deterministic(
        "CountrySPP_log_landings_yr",
        CountrySPP_log_landings[:,:,None]+CountrySPP_log_yr,
        dims=("exporter", "species", "year")
    )

    # National reliability
    lsd_intercept_sd_FAO = pm.Exponential("lsd_intercept_sd_FAO", 1)
    lsd_intercept_FAO = pm.Normal("lsd_intercept_FAO", sigma=lsd_intercept_sd_FAO)

    lsd_unreliability_sd_FAO = pm.Exponential("lsd_unreliability_sd_FAO", 1)
    lsd_unreliability_effect_FAO = pm.Normal(
        #"lsd_unreliability_effect_FAO", sigma=lsd_unreliability_sd_FAO, dims="exporter"
        "lsd_unreliability_effect_FAO", sigma=lsd_unreliability_sd_FAO
    )

    reliability_FAO = pyt.exp(
        lsd_intercept_FAO + lsd_unreliability_effect_FAO * unreliability_data
    )
    
    # Likelihood
    pm.Normal(
        "obs_spp",
        CountrySPP_log_landings_yr[country_spp_id, species_spp_id, year_spp_id],
        reliability_FAO[country_spp_id],
        observed=logReported_species_landings,
    )

    # = = = = = = = = = = = = = = = = TAXON MODEL = = = = = = = = = = = = = = = = #

    # National average landings by species in taxon branch - remainder after species allocation
    CountrySPP_Taxonlandings = pm.Deterministic(
        "CountrySPP_Taxonlandings",
        (Latent_landings-CountrySPP_landings)*NoTaxAgg[:,None],
        dims=("exporter", "species"),
    )

    # Scale taxon aggregations
    taxon_intercept = pm.Normal('taxon_intercept', np.log(3), 2)
    taxon_numspp = pm.Normal('taxon_numspp')
    taxon_magnitude = pm.Deterministic('taxon_magnitude', pyt.exp(taxon_intercept+taxon_numspp*np.log(ntaxon_country+0.0001)) )
    # log-odds for species contributions to taxon aggregations
    TaxonAGG_odds_ = pm.Normal("TaxonAGG_odds_", -1*TaxonMASK_Sx[TaxonMASK_Sx == 1], taxon_magnitude[TaxonMASK_Sx == 1])
    # Expand to full rank - make impossibles zero for adding in -999 at next step
    TaxonAGG_odds = pm.Deterministic(
        "TaxonAGG_odds",
        pyt.set_subtensor((pyt.ones(TaxonMASK_Sx.shape)*-9)[TaxonMASK_Sx == 1], TaxonAGG_odds_),
        dims=("exporter", "species", "taxon"),
    )

    # log-odds for species contributions to taxon aggregations
    TaxonAGG_odds2_ = pm.Normal("TaxonAGG_odds2_", -1*TaxonMASK_Sx[TaxonMASK_Sx == 1], 2)
    # Expand to full rank - make impossibles zero for adding in -999 at next step
    TaxonAGG_odds2 = pm.Deterministic(
        "TaxonAGG_odds2",
        pyt.set_subtensor((pyt.ones(TaxonMASK_Sx.shape)*-9)[TaxonMASK_Sx == 1], TaxonAGG_odds2_),
        dims=("exporter", "species", "taxon"),
    )

    # Softmax for allowable taxa in country - CAN ADD COVARIATES FOR TAXON CLASSIFICATION HERE IF EVER AVAILABLE; TaxonAGG_odds need
    # be pyt.zeros and the refined step used if covariates added
    #TaxonAGG_odds_refined = TaxonMASK_NEG*10 + TaxonAGG_odds
    # Binomial probability for 2 reporting categories
    TaxonAGG = pm.math.invlogit(TaxonAGG_odds2)
    # Three probabilities for >3, 2, or <=1 reported taxa
    P_TaxonSPP = pm.Deterministic(
        "P_TaxonSPP",
        # Softmax for exporters with 3 or more taxon aggregation categories; 0.4 centres proportions
        pm.math.softmax(TaxonAGG_odds, axis=2)*taxindx3[:,None,None]
        # Invlogit for exporters with 2 taxon aggregation categories - log sum to ensure unity
        + pyt.exp(pyt.log(TaxonAGG)-pyt.log((pyt.sum(TaxonAGG, axis=2))[:, :, None]))*taxindx2[:,None,None]
        # Unity for exporters reporting only one aggregation category
        + TaxonMASK_t1,
        dims=("exporter", "species", "taxon"),
    )

    # Latent species within taxa
    CountryTaxon_SPP_landings = pm.Deterministic(
        "CountryTaxon_SPP_landings",
        P_TaxonSPP*CountrySPP_Taxonlandings[:, :, None],
        dims=("exporter", "species", "taxon"),
    )

    # National average reported log landings by taxon
    CountryTaxon_log_landings = pm.Deterministic(
        "CountryTaxon_log_landings",
        pyt.log(CountryTaxon_SPP_landings.sum(1)+0.0001),
        dims=("exporter", "taxon"),
    )

    # Allow for annual landings variation
    CountryTaxon_log_yr = pm.ZeroSumNormal('CountryTaxon_log_yr',dims=("exporter", "taxon", "year"),n_zerosum_axes=2)

    # Add on extra bit
    CountryTaxon_log_landings_yr = pm.Deterministic(
        "CountryTaxon_log_landings_yr",
        CountryTaxon_log_landings[:,:,None]+CountryTaxon_log_yr,
        dims=("exporter", "taxon", "year")
    )

    # Likelihood
    pm.Normal(
        "obs_taxon",
        CountryTaxon_log_landings_yr[country_tax_id, taxon_tax_id, year_tax_id],
        reliability_FAO[country_tax_id],
        observed=logReported_taxon_landings,
    )
    #"""


# In[9]:


with landings_model_x:
    #pm.sample(draws=50, tune=200, trace=ch_backend, idata_kwargs=dict(log_likelihood=False))
    idata_landings_x = pm.sample(draws=400, tune=400, idata_kwargs=dict(log_likelihood=False))
print('Done sampling')


# In[ ]:


# Sample from prior and posterior predictive distributions
with landings_model_x:
    try:
        idata_landings_x.extend(pm.sample_prior_predictive())
        idata_landings_x.extend(pm.sample_posterior_predictive(idata_landings_x))
        print('Done predictives')
    except:
        print('Failed predictives')


# In[ ]:


"""
# ArviZ doesn't handle MultiIndex yet
# Making it aware of the real data labeling at the obs level
biggest_countries_long = kdata.country_name_abbreviation[
    [list(kdata.iso_3digit_alpha).index(x) for x in biggest_countries]
].to_numpy()

"""
obs_idx_detailed = tdata_cut_sharks.set_index(["ISOex_i", "ISOim_j"]).index

more_coords = {
    "obs_idx": obs_idx_detailed,
    "ISOex_i": ("obs_idx", tdata_cut_sharks["ISOex_i"]),
    "ISOim_j": ("obs_idx", tdata_cut_sharks["ISOim_j"]),
}
"""

more_coords = {
    "Exporter": ("exporter", biggest_countries_long),
    "Importer": ("importer", biggest_countries_long)
}


idata_landings_x.prior = idata_landings_x.prior.assign_coords(more_coords)
idata_landings_x.prior_predictive = idata_landings_x.prior_predictive.assign_coords(more_coords)
idata_landings_x.posterior = idata_landings_x.posterior.assign_coords(more_coords)
idata_landings_x.posterior_predictive = idata_landings_x.posterior_predictive.assign_coords(
    more_coords
)
idata_landings_x.observed_data = idata_landings_x.observed_data.assign_coords(more_coords)
idata_landings_x.constant_data = idata_landings_x.constant_data.assign_coords(more_coords)
#"""


# In[ ]:


# Export results
idata_landings_x.to_netcdf("idata-landings-model_AUGM.nc")
print("Saved trace!")


# In[ ]:


# Grab species level estimates for observed data
Obs_spp_data = np.exp(sdata.drop(columns=['year','year_spp_id','country_spp_id','species_spp_id']
          ).groupby(['country','species']).mean()).rename(columns={"logReported_species_landings": "Reported_landings"})

# Create xarray data of totals by species
Obs_spp_sums = xr.Dataset(Obs_spp_data.groupby(level='species').sum().sort_values(by='Reported_landings', ascending=False))

# Grab taxon level estimates for observed data
Obs_tax_data = np.exp(txdata.drop(columns=['year','year_tax_id','country_tax_id','taxon_tax_id']
          ).groupby(['country','taxon']).mean()).rename(columns={"logReported_taxon_landings": "Reported_landings"})

# Net estimated unreported species-level landings
Net_spp_landings = idata_landings_x.posterior['Latent_landings']-idata_landings_x.posterior['CountrySPP_landings']
Net_spp_landings_tax = idata_landings_x.posterior['CountrySPP_Taxonlandings']


# In[ ]:


#"""
mu_pp = idata_landings_x.posterior_predictive["obs_spp"]
_, ax = plt.subplots()
ax.scatter(idata_landings_x.observed_data["obs_spp"], mu_pp.mean(("chain", "draw")))
az.plot_hdi(idata_landings_x.observed_data["obs_spp"], mu_pp)
ax.plot((0,12.5),(0,12.5),linestyle=":")
ax.set_xlabel("Observed")
ax.set_ylabel("Expected")
ax.set_title("Observed vs Expected log Species landings")
plt.savefig(bf+'/Diagnostics/'+'Observed_vs_Expected_log_shark_trade.jpg',dpi=300);
#"""


# In[ ]:


#"""
mu_pp = idata_landings_x.posterior_predictive["obs_taxon"]
_, ax = plt.subplots()
ax.scatter(idata_landings_x.observed_data["obs_taxon"], mu_pp.mean(("chain", "draw")))
az.plot_hdi(idata_landings_x.observed_data["obs_taxon"], mu_pp)
ax.plot((0,12.5),(0,12.5),linestyle=":")
ax.set_xlabel("Observed")
ax.set_ylabel("Expected")
ax.set_title("Observed vs Expected log Taxon landings")
plt.savefig(bf+'/Diagnostics/'+'Observed_vs_Expected_log_shark_trade.jpg',dpi=300);
#"""


# In[ ]:


# = = = = = Set up plot
_, ax = plt.subplots(1,2,figsize=(10, 15))

# Select top 10% of landed species
tmp = idata_landings_x.posterior['CountrySPP_landings'].sum(('exporter')).mean(('chain','draw'))
tmp = tmp.sortby(tmp, ascending=False)
sppx_landed = tmp[tmp>np.quantile(tmp,0.90)].species

# Select top 10% of unreported species
tmp = Net_spp_landings.sum(('exporter')).mean(('chain','draw'))
tmp = tmp.sortby(tmp, ascending=False)
sppx_unrep = tmp[tmp>np.quantile(tmp,0.90)].species

# = = = = = Plot reported species landings
az.plot_forest(idata_landings_x.posterior['CountrySPP_landings'].sum(('exporter')).sel(species=sppx_landed).rename(''),
                    hdi_prob=0.9,
                    #transform=np.exp,
                    #figsize=(10, 20)
               ax=ax[0]
                   )
ax[0].set(
    title="Top reported landings",
    xlabel="tonnes",
    ylabel="",
);
#ax[0].set_xlim(0,200000)
for tick in ax[0].get_xticklabels():
    tick.set_rotation(45)

# = = = = = Plot unreported species landings
az.plot_forest(Net_spp_landings.sum(('exporter')).sel(species=sppx_unrep).rename(''),
                    #transform=np.exp,
                    #figsize=(10, 20)
                    hdi_prob=0.9,
                   ax=ax[1]
                   )
ax[1].set(
    title="Top unreported landings",
    xlabel="tonnes",
    ylabel="",
)
#ax[1].set_xlim(0,120000)
for tick in ax[1].get_xticklabels():
    tick.set_rotation(45)

plt.savefig(bf+'/Global/'+'Reported_Unreported_species.jpg',dpi=300);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




