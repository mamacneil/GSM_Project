#!/usr/bin/env python
# coding: utf-8

# # Model plots for shark and ray meat landings and trade applied to 2012-2019 data

# In[55]:


#!/usr/bin/env python
# coding: utf-8

# # Model for shark and ray meat landings and trade applied to 2014-2019 data

# In[1]:


import os
import pdb

import arviz as az
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pyt
import rdata as rd
import seaborn as sns
import xarray as xr
import scipy as sp
from matplotlib.gridspec import GridSpec

# Set figure style.
az.style.use("arviz-darkgrid")
bd = os.getcwd() + "/../Data/"

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


def unique(series: pd.Series):
    "Helper function to sort and isolate unique values of a Pandas Series"
    return series.sort_values().unique()


# ## Load landings data

# In[56]:


# In[83]:

#'''
dnam = bd + "modeldatsharksrays_land_trade_2012-2019"
parsed = rd.parser.parse_file(dnam + ".RData")
converted = rd.conversion.convert(parsed)
tmp = converted["modeldat"]
#'''

#'''
# Matrices
speciesCountryIDMap = (
    tmp["speciesCountryIDMap"]
    .rename({"dim_0": "species", "dim_1": "country", "dim_2": "taxon"})
    .transpose("country", "species", "taxon")
    .sortby("country")
)

logProbPrior = (
    tmp["logProbPrior"]
    .rename({"dim_0": "species", "dim_1": "country", "dim_2": "taxon"})
    .transpose("country", "species", "taxon")
    .sortby("country")
)

priorImportance = (
    tmp["priorImportance"]
    .rename({"dim_0": "species", "dim_1": "country"})
    .transpose("country", "species")
    .sortby("country")
)

OutputSpeciesCountryMapFull = (
    tmp["OutputSpeciesCountryMapFull"]
    .rename({"dim_0": "species", "dim_1": "country"})
    .transpose("country", "species")
    .sortby("country")
)

OutputSpeciesCountryMap = (
    tmp["OutputSpeciesCountryMap"]
    .rename({"dim_0": "species", "dim_1": "country"})
    .transpose("country", "species")
    .sortby("country")
)

SpeciesCommodityMap = (
    tmp["speciesCommodities"]
    .rename({"dim_0": "species", "dim_1": "commodity"})
    .sortby("species")
)

# Grab key for shark/ray
srkey = pd.read_csv(bd + "taxonomy_20240205.csv")
srkey['group'] = srkey.Superorder.replace('Batoidea','rays').replace('Selachimorpha','sharks').to_numpy()

# Trade covariate
tradeImportance = (
    tmp["tradeImportance"]
    .rename({"dim_0": "species", "dim_1": "country"})
    .transpose("country", "species")
    .sortby("country")
)


# ## Ensure available taxon matches for reported species

# In[57]:


#"""
# Initialize species to taxon mapping
SpeciesTaxonMAP = speciesCountryIDMap[0].drop_vars('country')

# Match taxa to species level regardless of taxonomic level of aggregation
for t in speciesCountryIDMap.taxon.values:
    # Iterate over possible species for each taxon
    for s in srkey[srkey.isin([t]).any(axis=1)]['species_binomial'].values:
        try:
            SpeciesTaxonMAP.loc[dict(species=s,taxon=t)] = 1
        except:
            pass

# List of rarely caught species 
drop_spp = priorImportance.species.to_numpy()[priorImportance.max(["country"])<=1]
# Number of species remaining with prior importance less than or equal to 1
priorImportance.species.shape[0]-drop_spp.shape[0]
# Make temporary list of all taxon IDs
tmp_taxon_ = speciesCountryIDMap.taxon.to_numpy()
# Index taxons relative to what gets landed
tmp_TaxonIDx = match(tmp["LandingsID"], list(tmp_taxon_))
# Temporary list of all species IDs
tmp_species_ = speciesCountryIDMap.species.to_numpy()
# Boolean of taxons that are to species level in observed landings
tmp_tindx = pd.Series(tmp_taxon_[tmp_TaxonIDx]).str.count(" ").to_numpy() == 0
# Index of species in taxon that are observed as catches
tmp_species_spp_id = match(tmp_taxon_[tmp_TaxonIDx[tmp_tindx == 0]], list(tmp_species_))
# Unique names of species in taxon that are observed as taxon catches but have prior importance <=1
tmp_spp = np.unique(tmp_species_[tmp_species_spp_id[np.log1p(tmp["allCatch"])[tmp_tindx == 0]>0]])
# Species in drop list that are actually observed as in taxon+species (taxon) list
tmp_spp = list(drop_spp[np.array([x in tmp_spp for x in drop_spp])])

# Grab landings data to see which taxons have catch
tmp_landings = tmp["allCatch"]
tmp_taxon = tmp["LandingsID"]
tmp_country = tmp["country"]
# Iterate over landings to ensure species are avaiable for taxon in country
for l,t,c in zip(tmp_landings,tmp_taxon,tmp_country):
    # Look for species landed with impossible priors
    try:
        if (priorImportance.sel(country=c,species=t)<=-888)*(l>0):
            priorImportance.loc[dict(country=c,species=t)] = 2
            tmp_spp += [t]
            #print("Changed "+c+" "+t)
    # Look for taxon landed with no possible species 
    except:
        # Possible species for taxon
        tax_spp = (SpeciesTaxonMAP.sel(taxon=t).species[SpeciesTaxonMAP.sel(taxon=t)==1]).values
        # If all species are impossible for taxon
        if (priorImportance.sel(country=c,species=tax_spp).mean()<=-888)*(l>0):
            # Assign to possible species in nation
            if OutputSpeciesCountryMapFull.sel(country=c,species=tax_spp).sum()>0:
                axx = tax_spp[OutputSpeciesCountryMapFull.sel(country=c,species=tax_spp).to_numpy()>0]
                xflax = 'ok'
            else:
                # Assign to most likely spp given global max
                axx = tax_spp[(priorImportance.sel(species=tax_spp).max('country')==priorImportance.sel(species=tax_spp).max())]
                xflax = 'not present'
            priorImportance.loc[dict(country=c,species=axx)] = 2
            tmp_spp += list(axx)
            #print("Impossible "+t+" changed "+c+", spp are "+xflax)
            #print(axx)
        
        # If all species for taxon are below data-reduction cutoff
        elif (priorImportance.sel(country=c,species=tax_spp).max()<2)*(l>0):
            # Assign to possible species in nation
            if OutputSpeciesCountryMapFull.sel(country=c,species=tax_spp).sum()>0:
                axx = tax_spp[OutputSpeciesCountryMapFull.sel(country=c,species=tax_spp).to_numpy()>0]
                xflax = 'ok'
            else:
                # Assign to most likely spp given global max
                axx = tax_spp[(priorImportance.sel(species=tax_spp).max('country')==priorImportance.sel(species=tax_spp).max())]
                xflax = 'not present'
            priorImportance.loc[dict(country=c,species=axx)] = 2
            tmp_spp += list(axx)
            #print("Low "+t+" changed "+c+", spp are "+xflax)
            #print(axx)

# Add in species in field data
field_tmp = np.array(['Atlantoraja cyclhophora', 'Bathtoshia centroura',
       'Callorynchus callorynchus',
       'Dasyatis hypostigma', 'Fontitrygon geijskesi',
       'Hypanus bethalutzae', 'Mobula hypostoma', 'Narcine brasiliensis',
       'Pseudobatos horkelii', 'Pteroplatytrygon violacae',
       'Rhinoptera brasilisensis', 'Rioraja agassizi',
       'Scyliorhinus haekelii', 'Squalus albicaudus', 'Squatina occulta',
       'Zapteryx brevirostris'])
for s in field_tmp:
    if s not in tmp_spp:
        tmp_spp+=[s]

# Species to drop = have prior importance <=1 AND are not actually observed as taxon to the species level
drop_spp = drop_spp[np.array([x not in tmp_spp for x in drop_spp])]

#"""
# Drop rare and unreported species
speciesCountryIDMap = speciesCountryIDMap.drop_sel(species=drop_spp)
priorImportance = priorImportance.drop_sel(species=drop_spp)
logProbPrior = logProbPrior.drop_sel(species=drop_spp)
SpeciesCommodityMap = SpeciesCommodityMap.drop_sel(species=drop_spp)
tradeImportance = tradeImportance.drop_sel(species=drop_spp)

# = = = = = = = = = = = = = = After species drop = = = = = = = = = = = = = #

# Vectors
allCatch = tmp["allCatch"]
cindx = allCatch>0
allCatch = allCatch[cindx]
logCatch = np.log1p(tmp["allCatch"])[cindx]
species_ = logProbPrior.species.to_numpy()
country_ = logProbPrior.country.to_numpy()
CountryIDx = match(tmp["country"][cindx], list(country_))
year_ = ["year_" + str(x) for x in np.unique(tmp["year"][cindx]).astype(int)]
YearIDx = match(tmp["year"][cindx], list(np.unique(tmp["year"][cindx]).astype(int)))
taxon_ = logProbPrior.taxon.to_numpy()
TaxonIDx = match(tmp["LandingsID"][cindx], list(taxon_))
speciesCountryMap = speciesCountryIDMap.groupby("species").max("taxon")
TaxonPRIOR = priorImportance.to_numpy()

# Meat species
#meat_mask = SpeciesCommodityMap.sel(commodity='meat',species=species_).to_numpy()
meat_mask = 1*(SpeciesCommodityMap.sel(commodity=('fins'),species=species_)+SpeciesCommodityMap.sel(commodity=('meat'),species=species_)>0).to_numpy()
meat_mask[meat_mask==0] = -999
meat_mask[meat_mask==1] = 0


# ## Set up split data

# In[58]:


# Calculate average total catch for target of softmax
TotalCatch = np.array([allCatch[CountryIDx==i].sum() for i in range(speciesCountryIDMap.shape[0])])/len(year_)
logTotalCatch = np.log(TotalCatch)


# In[60]:


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

# Make group vector
group_ = srkey.group[match(species_,list(srkey.species_binomial))].to_numpy()
# Make masks for trade sums
shark_mask = 1*(group_=='sharks')
ray_mask = 1*(group_=='rays')

# Grab only aggregated taxa
taxon_shortlist = taxon_[pd.Series(taxon_).str.count(" ").to_numpy() == 0]

# Split data index
tindx = pd.Series(taxon_[TaxonIDx]).str.count(" ").to_numpy() == 0
# Split landings
logReported_species_landings = logCatch[tindx == 0]
logReported_taxon_landings = logCatch[tindx == 1]
# Split country index
country_spp_id = CountryIDx[tindx == 0]
country_tax_id = CountryIDx[tindx == 1]
# Split year index
year_spp_id = YearIDx[tindx == 0]
year_tax_id = YearIDx[tindx == 1]
# Split species index
species_spp_id = match(taxon_[TaxonIDx[tindx == 0]], list(species_))
# Split taxon index
# taxon_tax_id = TaxonIDx[tindx==1]
taxon_tax_id = match(taxon_[TaxonIDx[tindx == 1]], list(taxon_shortlist))



# Make dataframes for later
sdata = pd.DataFrame(
    {
        "logReported_species_landings": logReported_species_landings,
        "species_spp_id": species_spp_id,
        "country_spp_id": country_spp_id,
        "year_spp_id": year_spp_id,
        "country": country_[country_spp_id],
        "year": np.array(year_)[year_spp_id],
        "species": species_[species_spp_id],
    }
)
txdata = pd.DataFrame(
    {
        "logReported_taxon_landings": logReported_taxon_landings,
        "taxon_tax_id": taxon_tax_id,
        "country_tax_id": country_tax_id,
        "year_tax_id": year_tax_id,
        "country": country_[country_tax_id],
        "year": np.array(year_)[year_tax_id],
        "taxon": taxon_shortlist[taxon_tax_id],
    }
)
txdata = txdata.loc[match(txdata.taxon, list(taxon_shortlist)) != None]


# ## Set up masking

# In[61]:


# Initial mask of species matched to possible taxon
InitTaxonMASK = speciesCountryIDMap.copy()
# Create empty mask to store observed taxa as possible
TaxonMASK = InitTaxonMASK*0


# In[62]:


# Reported taxa by country
Obs_tax_data = np.exp(txdata.drop(columns=['year','year_tax_id','country_tax_id','taxon_tax_id']
          ).groupby(['country','taxon']).mean()).rename(columns={"logReported_taxon_landings": "Reported_landings"})


# In[63]:


# Iteratre over countries
for c in country_:
    try:
        # Iterate over observed taxa and make possible
        taxes = Obs_tax_data.loc[c].index.values
        for t in taxes:
            for s in species_:
                TaxonMASK.loc[dict(country=c,species=s,taxon=t)]=InitTaxonMASK.loc[dict(country=c,species=s,taxon=t)]
    except:
        #print(c)
        pass


# In[64]:


# Make priors using relative odds to proportion total landings
SppPRIOR = priorImportance.to_numpy()
SppPRIORadj = SppPRIOR.copy()
# Re-weight to log-odds scale
SppPRIORadj[SppPRIORadj==-2]=-4.5
SppPRIORadj[SppPRIORadj==-1]=-3.5
SppPRIORadj[SppPRIORadj==0]=-2.5
SppPRIORadj[SppPRIORadj==1]=-1
SppPRIORadj[SppPRIORadj==2]=2
SppPRIORadj[SppPRIORadj==3]=5

# Cut down taxon MASK to match taxon_shortlist dimensions
#TaxonMASK_S = TaxonMASK[:, :, match(taxon_shortlist, list(taxon_))]
TaxonMASK_S = TaxonMASK.sel(taxon=taxon_shortlist)
TaxonMASK_Sx = TaxonMASK_S.to_numpy()

# Negative mask for log-odds zeros
negval = -9
TaxonMASK_NEG = TaxonMASK_S.copy().to_numpy()
TaxonMASK_NEG[TaxonMASK_NEG==0] = negval


# Species weight for countries with no aggregations - huge log-odds so p(species ID)=1 where needed
NoTaxaSppWT = np.zeros(SppPRIOR.shape)
NoTaxaSppWT[(TaxonMASK_NEG != negval).sum(1).sum(1) == 0, :] = abs(negval)


# In[65]:


tmp = np.array(['Carcharhinus acronotus', 'Carcharhinus brevipinna',
       'Dasyatis hypostigma', 'Heptranchias perlo',
       'Narcine brasiliensis', 'Rhinoptera bonasus',
       'Rhizoprionodon lalandii'])


# ## Match landings and trade data

# In[66]:


# Make fdata table to merge with trade model
fdata = pd.DataFrame(
    {
        "year": YearIDx + 2012,
        "country_code": country_[CountryIDx],
        "species": taxon_[TaxonIDx],
        "landed_weight": allCatch,
    }
)

# Add shark/ray group for each taxon in landings
tmp_taxon = fdata.species.unique()
tmp_group = []

for tx in tmp_taxon:
    # taxon at species level
    if tx in srkey.species_binomial.to_numpy():
        tmp_group += [srkey.group[srkey.species_binomial==tx].values[0]]
    elif tx in srkey.Genus.to_numpy():
        tmp_group += [srkey.group[srkey.Genus==tx].values[0]]
    elif tx in srkey.Family.to_numpy():
        tmp_group += [srkey.group[srkey.Family==tx].values[0]]
    elif tx in srkey.Order.to_numpy():
        tmp_group += [srkey.group[srkey.Order==tx].values[0]]
    elif tx in ['Sphyrnidae','Selachimorpha']:
        tmp_group += ['sharks']
    elif tx in ['Elasmobranchii']:
        tmp_group += ['elasmos']
    else:
        print(tx)
fdata['group'] = np.array(tmp_group)[match(fdata.species,list(tmp_taxon))]



## CHECK THAT ADDITIONAL ELASMOS ARE OK WITH RE-EXPORT CALCULATIONS. 
## CHECK ALL LANDINGS AND TRADE DATA TO ENSURE SAME YEAR-LEVEL OBSERVATIONS


# ## Import commodity code table

# Import taxonomic match table for BACI commodity codes and species (MASK)
cdata = pd.read_csv(bd + "comm.code.taxon.match.csv")

# ## Load BACI keys

# Import BACI commodity code key
ckey = pd.read_csv(bd + "product_codes_HS12_V202102.csv")
ckey17 = pd.read_csv(bd + "product_codes_HS17_V202102.csv")

# Import BACI country keys
kdata = pd.read_csv(bd + "country_codes_V202301.csv")
kdata.country_code = kdata.country_code.values.astype(int)

# TWN doesn't have an ISO code
kdata.loc[
    kdata.country_name_full == "Other Asia, not elsewhere specified", "iso_3digit_alpha"
] = "TWN"


# ## Load BACI seafood trade

# Import overall trade from BACI data
odata = pd.read_csv(bd + "baci.seafood_12-19_ij_all.csv")

# Make them numeric
odata.exporter_i = odata.exporter_i.values.astype(int)
odata.importer_j = odata.importer_j.values.astype(int)

# Add country codes

odata["ISOex_i"] = kdata.iso_3digit_alpha.values[
    match(list(odata.exporter_i.values), list(kdata.country_code.values))
]
odata["ISOim_j"] = kdata.iso_3digit_alpha.values[
    match(list(odata.importer_j.values), list(kdata.country_code.values))
]


# ## Load BACI meat trade

# Import BACI data
#tdata = pd.read_csv(bd + "baci.elasmo_HS12_2012-2017.csv")
tdata = pd.read_csv(bd + "baci_HS12-19_elasmo.csv")
tdata = tdata[tdata['quantity_q'].notna()]
tdata = tdata[tdata['quantity_q']!='           NA']

# Make them numeric
tdata.exporter_i = tdata.exporter_i.values.astype(int)
tdata.importer_j = tdata.importer_j.values.astype(int)
tdata.quantity_q = tdata.quantity_q.values.astype(float)

# Temporary change of code for Italy
tdata.loc[tdata.exporter_i == 381, "exporter_i"] = 380
tdata.loc[tdata.importer_j == 381, "importer_j"] = 380

# Add country names for imports/exports
tdata["ISOex_i"] = kdata.iso_3digit_alpha.values[
    match(list(tdata.exporter_i.values), list(kdata.country_code.values))
]
tdata["ISOim_j"] = kdata.iso_3digit_alpha.values[
    match(list(tdata.importer_j.values), list(kdata.country_code.values))
]

# Add explicit code descriptions
tdata['hs_description'] = ckey17.description.values[match(tdata["hscode_k"],list(ckey17.code.values))]

# Make fresh/frozen fins dummy
tmp = np.array([0]*len(tdata['hs_description']))
tmp[tdata["hscode_k"]==30292] = 1
tmp[tdata["hscode_k"]==30392] = 1
tdata['fins'] = tmp

# Make rays dummy
tdata['rays'] = np.array(['rays' in l for l in tdata['hs_description']])*1
# Make 30488 sharks
tdata.rays[tdata.hscode_k==30488] = 0

# Add commodity group
tmp = np.array(['sharks']*len(tdata['rays']))
tmp[tdata['rays']==1] = 'rays'
tdata['group'] = tmp


# In[67]:


# Grab sharks in years that have fresh/frozen fins separated
tdata_17 = tdata.copy()[(tdata.year_t>=tdata[tdata.fins==1].year_t.min()) & (tdata.group=='sharks')]


# In[68]:


# Get total exports of fins and meat per year, exporter, importer combo
tmp = tdata_17.groupby(["ISOex_i","ISOim_j","year_t","fins"]).sum().reset_index()
# Grab fins only
tmp_fins = tmp.copy()[tmp.fins==1]
# Sum over total per exporter, importer combo
tmp_sum = tmp[["ISOex_i","ISOim_j","quantity_q"]].groupby(["ISOex_i","ISOim_j"]).sum()
# Sum fins over total per year, exporter, importer
tmp_fins_sum = tmp_fins[["ISOex_i","ISOim_j","quantity_q"]].groupby(["ISOex_i","ISOim_j"]).sum()
# Grab proportion of fins where there are any
tmp_fin_props = (tmp_fins_sum/tmp_sum).reset_index()
# Make zero fins where there are none
tmp_fin_props.quantity_q[tmp_fin_props.quantity_q.isna()] = 0
# Convert to proportions of meat
tmp_meat_props = tmp_fin_props.copy()
tmp_meat_props['prop_sharks'] = 1-tmp_meat_props.quantity_q
tmp_meat_props = tmp_meat_props.drop(columns=['quantity_q'])


# In[69]:


# Plot average proportions of shark exports that are fins
#plt.hist(tmp_fin_props.quantity_q[tmp_fin_props.quantity_q>0]);


# In[70]:


tdata_old = tdata.copy()

# Iterate over exporter/importer combos
for i in range(tmp_meat_props.shape[0]):
    exi = tmp_meat_props.ISOex_i[i]
    imi = tmp_meat_props.ISOim_j[i]
    shark_prop = tmp_meat_props.prop_sharks[i]
    # Re-scale observed sharks pre 2017 to remove potential fins
    tdata.quantity_q[(
        (tdata.year_t<2017) &
        (tdata.ISOex_i==exi) &
        (tdata.ISOim_j==imi)&(tdata.group=='sharks'))] = tdata.quantity_q[(
            (tdata.year_t<2017) &
            (tdata.ISOex_i==exi) &
            (tdata.ISOim_j==imi)&(tdata.group=='sharks'))]*shark_prop


# In[71]:


#plt.scatter(tdata_old.quantity_q,tdata.quantity_q)
#plt.xlabel('All shark trade')
#plt.ylabel('Adjusted shark trade')
#plt.title('Shark trade with fins removed 2012-2017');


# In[72]:


# Convert to live weights
tdata['estimated_live_weight'] = tdata.quantity_q
tdata.estimated_live_weight[tdata.group=='sharks'] = tdata.estimated_live_weight[tdata.group=='sharks']*2
tdata.estimated_live_weight[tdata.group=='rays'] = tdata.estimated_live_weight[tdata.group=='rays']*1.6


# In[73]:


#plt.hist(tdata[tdata.hscode_k==30488].estimated_live_weight);


# In[74]:


### Reset biggest_countries to include only those with landings
biggest_countries = country_

# - - - - - - - - - - - Add BACI total seafood trade
total_seafood_trade = (
    odata[
        ((odata.ISOex_i).isin(biggest_countries))
        & ((odata.ISOim_j).isin(biggest_countries))
    ]
    .groupby(["ISOex_i", "ISOim_j"])
    .sum()["ij_total"]
    .reset_index()
    .set_index("ISOex_i")
    .pivot(columns="ISOim_j")
    .droplevel(0, axis="columns")
    .fillna(0.0)
)


# ## Deal with re-exports

# In[75]:


# # REEXPORTS - NEED TO CHECK THAT RAYS ARE COOL HERE TOO
#
# Currently removes trade that has no possible catch.
#
# NB:
#
# 1. Assumes catches in year t are traded in year t

# Pre-removals copy
tdata_copy = tdata.copy()

# Empty list of identified re-exports
tmp = []
# Unique country list
tmp_c = np.unique(np.array(list(tdata.ISOex_i)+list(tdata.ISOim_j)))
# Unique commodity codes
tmp_u = tdata.group.unique()
# Landings per country per year per commodity
tmp_l = fdata.groupby(["country_code","year","group"]).sum().reset_index().sort_values("landed_weight", ascending=False)

# ====================== Remove re-exports from trade =========================== #
# Iterate over years
for y in tdata.year_t.unique():
    # Grab values for year y 
    trad_ = tdata[tdata.year_t==y].groupby(["ISOex_i",'year_t','hscode_k']).sum().reset_index().sort_values("estimated_live_weight", ascending=False)
    # Grab total trade
    trad = trad_.groupby(["ISOex_i"]).sum().reset_index().sort_values("estimated_live_weight", ascending=False)
    # Grab group trade
    trad_s = trad[trad.group=='sharks']
    trad_r = trad[trad.group=='rays']
    # Grab total landings for year y
    land = tmp_l[tmp_l.year==y]
    # Grab possible group landings for year y
    land_s = land[land.group=='sharks']
    land_r = land[land.group=='rays']
    land_e = land[land.group=='elasmos']
    # Iteratre over countries
    for e in tmp_c:
        if e in land.country_code.unique() and e in trad.ISOex_i.unique():
            # Grab values for exporter e in year y
            tx = trad[trad.ISOex_i==e]
            tx_s = trad_s[trad_s.ISOex_i==e]
            tx_r = trad_r[trad_r.ISOex_i==e]
            # Grab landings
            lx = land[land.country_code==e]
            lx_s = land_s[land_s.country_code==e]
            lx_r = land_r[land_r.country_code==e]
            lx_e = land_e[land_e.country_code==e]

            # If no catches to support trade, make trade zero to remove re-exports
            # Do this first because no catches of any kind trump group specifics
            # justified because the project is about assigning catches within the trade to specific nations
            if sum(lx.landed_weight)==0 and sum(tx.estimated_live_weight)>0:
                tmp += [sum(tx.estimated_live_weight)]
                tdata.estimated_live_weight[(tdata.ISOex_i==e) & (tdata.year_t==y)] = 0

            # If no catches (group+elasmos) to support shark trade, make trade zero to remove re-exports: 
            if sum(lx_s.landed_weight+lx_e.landed_weight)==0 and sum(tx_s.estimated_live_weight)>0:
                tmp += [sum(tx_s.estimated_live_weight)]
                tdata.estimated_live_weight[(tdata.ISOex_i==e) & (tdata.year_t==y) & (tdata.group=='sharks')] = 0
            # If no catches (group+elasmos) to support ray trade, make trade zero to remove re-exports:
            if sum(lx_r.landed_weight+lx_e.landed_weight)==0 and sum(tx_r.estimated_live_weight)>0:
                tmp += [sum(tx_r.estimated_live_weight)]
                tdata.estimated_live_weight[(tdata.ISOex_i==e) & (tdata.year_t==y) & (tdata.group=='rays')] = 0
            # If trade more than catches, make proportional within allowable commodity codes
            elif sum(lx.landed_weight)<sum(tx.estimated_live_weight):
                tmp += [sum(tx.estimated_live_weight)]
                # Grab proportion
                rrx = sum(lx.landed_weight)/sum(tx.estimated_live_weight)
                # Re-scale trade to proportion of total landings
                tdata.estimated_live_weight[(tdata.ISOex_i==e) & (tdata.year_t==y) & (tdata.group=='sharks')] = tdata.estimated_live_weight[(tdata.ISOex_i==e) & (tdata.year_t==y) & (tdata.group=='sharks')]*rrx
                tdata.estimated_live_weight[(tdata.ISOex_i==e) & (tdata.year_t==y) & (tdata.group=='rays')] = tdata.estimated_live_weight[(tdata.ISOex_i==e) & (tdata.year_t==y) & (tdata.group=='rays')]*rrx
        else:
            tdata.estimated_live_weight[(tdata.ISOex_i==e) & (tdata.year_t==y)] = 0
            print(e,y)        

# ## Restrict landings, seafood trade and meat trade to biggest countries

# Original
tmp_x = (tdata_copy.estimated_live_weight)
# Updated
tmp_y = (tdata.estimated_live_weight)
# Countries with trade reduced
iredux = (tdata_copy.estimated_live_weight-tdata.estimated_live_weight)>0
# Grap re-exports data
ReExports = pd.DataFrame(zip(tmp_x[iredux],tmp_y[iredux],tdata.ISOex_i[iredux].to_numpy(),tdata_copy.year_t[iredux],tdata_copy.group[iredux]),columns=['Original','Reduced','exporter','year','group'])
ReExports['Net_diff'] = ReExports.Original-ReExports.Reduced
ReExports['Exporter'] = kdata.country_name_abbreviation[[list(kdata.iso_3digit_alpha).index(x) for x in ReExports.exporter]].to_numpy()

# Table of re-exporting countries
tmp = ReExports.groupby(['Exporter']).sum().sort_values(by='Net_diff',ascending=False).drop(columns='exporter')
tmp.to_csv('ReExport_totals.csv')


# = = = = = = = = = = = = = #
# Remove zeros from trade
tdata = tdata[tdata.estimated_live_weight!=0]

# = = = = = = = = = = = = = #
# Remove fins
tdata = tdata[tdata['product']=='meat']


# ## Setup landings data

# In[76]:


# Split out shark and ray trade
tdata_cut = tdata[
    ((tdata.ISOex_i).isin(biggest_countries))
    & ((tdata.ISOim_j).isin(biggest_countries))
]
# dropping missing values
tdata_cut = tdata_cut.drop("Unnamed: 0", axis="columns").dropna().reset_index(drop=True)
# Summarize data by importer export commodity
tdata_cut = tdata_cut.groupby(["ISOex_i", "ISOim_j", "group"]).sum().reset_index()


# Get average per year
tdata_cut['estimated_live_weight'] = tdata_cut['estimated_live_weight']/len(tdata["year_t"].unique())

# we just care about whether it's a shark or ray, not if it's also fresh or frozen
tdata_cut["fish_type"] = tdata_cut["group"]

tdata_cut = tdata_cut.sort_values(["ISOex_i", "ISOim_j", "fish_type"])

# Shark data
tdata_cut_sharks = (
    tdata_cut[tdata_cut.fish_type == "sharks"]
    .groupby(["ISOex_i", "ISOim_j"])
    .sum()
    .reset_index()
)

# Ray data
tdata_cut_rays = (
    tdata_cut[tdata_cut.fish_type == "rays"]
    .groupby(["ISOex_i", "ISOim_j"])
    .sum()
    .reset_index()
)


# In[77]:


## Load unreliability score

unreliability_score = pd.read_csv(
    bd + "reporter_reliability_HS12_V202102.csv",
    usecols=["c", "q_unreliability_i", "q_unreliability_j"],
)

# ITA changed code -- because, why not?
unreliability_score.loc[unreliability_score.c == 381, "c"] = 380

# grab ISO codes from odata
unreliability_score = unreliability_score.rename(
    columns={"q_unreliability_i": "exporter", "q_unreliability_j": "importer"}
).merge(odata, left_on="c", right_on="exporter_i")
unreliability_score = (
    unreliability_score[
        ((unreliability_score.ISOex_i).isin(biggest_countries))
        & ((unreliability_score.ISOim_j).isin(biggest_countries))
    ][["ISOex_i", "exporter", "importer"]]
    .groupby("ISOex_i")
    .first()
)

# Check for missing unreliability scores
misx = biggest_countries[
    np.array([x not in unreliability_score.index for x in biggest_countries])
]

# Fill missing unreliablity scores with maximum unreliability value
tmp_maxval = max(unreliability_score.exporter)
if len(misx) > 0:
    for i in misx:
        unreliability_score = pd.concat(
            [
                pd.DataFrame(index=[i], columns=unreliability_score.columns),
                unreliability_score,
            ]
        )
unreliability_score = unreliability_score.sort_index().fillna(tmp_maxval)
unreliability_score = unreliability_score.reindex(
    sorted(unreliability_score.columns), axis=1
)


# ## Setup taxon masking

# In[78]:


# Count taxon aggregations
ntax_country = (
    txdata.groupby(by=(["country", "taxon"]))
    .sum()
    .reset_index()
    .groupby("country")
    .count()
    .taxon
)

# Add Belize
ntax_country["BLZ"] = 0
# Re-order to match country_
ntax_country = ntax_country[country_]

# Taxon by country groupings
taxindx1 = (ntax_country <= 1).to_numpy()
taxindx2 = (ntax_country == 2).to_numpy()
taxindx3 = (ntax_country >= 3).to_numpy()

# Create 3 dimensional mask
TaxonMASK_t1 = TaxonMASK_Sx.copy()
TaxonMASK_t2 = TaxonMASK_Sx.copy()
TaxonMASK_t3 = TaxonMASK_Sx.copy()

# Deactivate countries without <=1, 2, or >=3 taxon groups reported
TaxonMASK_t1[taxindx1 == False, ...] = 0
TaxonMASK_t2[taxindx2 == False, ...] = 0
TaxonMASK_t3[taxindx3 == False, ...] = 0

# Make sure Elasmos bin is positive in countries with no aggregations
NoTaxAgg = (NoTaxaSppWT.sum(1)==0)*1
TaxonMASK_Sx[NoTaxAgg!=1,:,list(taxon_shortlist).index('Elasmobranchii')] = 1


# ## Define `dims` and `coords`

# In[79]:


# some countries can be missing from importers or exporters
# indexing needs to take that into account
# if we used `factorize`, that would be ignored

country_to_idx_map = {country: index for index, country in enumerate(biggest_countries)}
shark_exporter_idx = tdata_cut_sharks["ISOex_i"].map(country_to_idx_map).to_numpy()
shark_importer_idx = tdata_cut_sharks["ISOim_j"].map(country_to_idx_map).to_numpy()
ray_exporter_idx = tdata_cut_rays["ISOex_i"].map(country_to_idx_map).to_numpy()
ray_importer_idx = tdata_cut_rays["ISOim_j"].map(country_to_idx_map).to_numpy()

# You have to be careful when creating shark_trade_matrix:
shark_trade_matrix = (
    tdata_cut_sharks[["ISOex_i", "ISOim_j", "estimated_live_weight"]]
    .set_index("ISOex_i")
    .pivot(columns="ISOim_j")
    .droplevel(0, axis="columns")
)

# Add missing exporters and importers
missing_col = []
for p in country_:
    if not p in shark_trade_matrix.columns.values:
        missing_col.append(p)
missing_col = np.array(missing_col)
shark_trade_matrix[missing_col] = np.NaN
shark_trade_matrix = shark_trade_matrix[country_]
missing_row = shark_trade_matrix.columns.difference(shark_trade_matrix.index)
shark_trade_matrix = shark_trade_matrix.T
shark_trade_matrix[missing_row] = np.NaN
shark_trade_matrix = shark_trade_matrix[country_]
shark_trade_matrix = shark_trade_matrix.T.sort_index().fillna(0)
shark_trade_mask = shark_trade_matrix.to_numpy()
shark_trade_mask[shark_trade_mask>0] = 1
# Add domestic consumption
np.fill_diagonal(shark_trade_mask,1)

# You have to be careful when creating ray_trade_matrix:
ray_trade_matrix = (
    tdata_cut_rays[["ISOex_i", "ISOim_j", "estimated_live_weight"]]
    .set_index("ISOex_i")
    .pivot(columns="ISOim_j")
    .droplevel(0, axis="columns")
)

# Add missing exporters and importers
#missing_col = ray_trade_matrix.index.difference(ray_trade_matrix.columns)
missing_col = []
for p in country_:
    if not p in ray_trade_matrix.columns.values:
        missing_col.append(p)
missing_col = np.array(missing_col)
ray_trade_matrix[missing_col] = np.NaN
ray_trade_matrix = ray_trade_matrix[country_]
missing_row = ray_trade_matrix.columns.difference(ray_trade_matrix.index)
ray_trade_matrix = ray_trade_matrix.T
ray_trade_matrix[missing_row] = np.NaN
ray_trade_matrix = ray_trade_matrix[country_]
ray_trade_matrix = ray_trade_matrix.T.sort_index().fillna(0)
ray_trade_mask = ray_trade_matrix.to_numpy()
ray_trade_mask[ray_trade_mask>0] = 1
# Add domestic consumption
np.fill_diagonal(ray_trade_mask,1)

# Species mask for possible trade (including domestic)
trade_mask = ray_trade_mask[:,None,:]*((group_=='rays')[None,:,None])+shark_trade_mask[:,None,:]*((group_=='sharks')[None,:,None])
trade_mask[trade_mask==0] = -999
trade_mask[trade_mask>0] = 0
# Remove species not used for meat
#trade_mask = trade_mask+meat_mask[None,:,None]
trade_mask[trade_mask<0] = -999

# Mask for trade softmax to zero out species with all -999
NoSPP_Mask = (((trade_mask==-999).sum(2)!=len(country_))*1)

# Mask for blue shark relative odds importer preferences
BSmask = np.zeros(shape=trade_mask[0].shape)
BSmask[list(species_).index('Prionace glauca')] = -999

# Better country labels
biggest_countries_long = kdata.country_name_abbreviation[
    [list(kdata.iso_3digit_alpha).index(x) for x in biggest_countries]
].to_numpy()

# Create matching tensor for priors
SppPRIORadj_idx = SppPRIORadj.copy()
# List of unique prior values
priors_ = list(np.sort(np.unique(SppPRIORadj)))
# Replace prior values with index to OddsCAT
for i in range(len(SppPRIORadj_idx)):
    SppPRIORadj_idx[i] = match(SppPRIORadj_idx[i],priors_)

COORDS = {
    "exporter": biggest_countries,
    "importer": biggest_countries,
    "shark_obs_idx": tdata_cut_sharks.index,
    "ray_obs_idx": tdata_cut_rays.index,
    "direction": ["exports", "imports"],
    "quantity": ["weight", "value"],
    "species": species_,
    "landing_country": country_,
    "taxon": taxon_shortlist,
    "year":year_,
    "OddsCAT":np.unique(SppPRIORadj).astype(str)
}
print("Data loaded!")


# In[ ]:




