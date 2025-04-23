#!/usr/bin/env python
# coding: utf-8

# # Model data for shark and ray meat landings and trade applied to 2012-2019 data

# In[1]:


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

# In[2]:


# In[83]:

#'''
#dnam = bd + "/fishorshark/modeldatsharksraysImportNeg1augmented"
dnam = bd + "modeldat_Augmented20250116_CHNupdates"
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

# Observed landings
ObsLandings_ = pd.melt(tmp["national_landings_info"], id_vars='country', value_name='landings')
ObsLandings_.columns = ['country','species','obs_landings']
ObsLandings_.species = np.array([s.replace('.' , ' ') for s in ObsLandings_.species.values])
ObsLandings_ = ObsLandings_.set_index(['country','species']).to_xarray().to_dataarray()[-1]


# In[ ]:





# In[3]:


# Create an empty DataFrame with the specified index and columns
scorez = list(np.unique(priorImportance).astype(int).astype(str))
tradeweightcounts = pd.DataFrame(index=priorImportance.country, columns=scorez).fillna(0)
for c in priorImportance.country.to_numpy():
    colz, valz = np.unique(priorImportance[priorImportance.country==c,:],return_counts=True)
    tradeweightcounts.loc[c,colz.astype(int).astype(str)] = valz


# In[4]:


tradeweightcounts_init = tradeweightcounts


# In[5]:


#tradeweightcounts_init


# ## Species checks

# In[6]:


priorImportance.loc[:,priorImportance.species=='Fontitrygon garouaensis'] = -999


# ## Ensure available taxon matches for reported species

# In[7]:


# Initialize species to taxon mapping
SpeciesTaxonMAP = speciesCountryIDMap[0].drop_vars('country')
#np.unique(SpeciesTaxonMAP, return_counts=True)


# In[8]:


# Match taxa to species level regardless of taxonomic level of aggregation
for t in speciesCountryIDMap.taxon.values:
    # Iterate over possible species for each taxon
    for s in srkey[srkey.isin([t]).any(axis=1)]['species_binomial'].values:
        try:
            SpeciesTaxonMAP.loc[dict(species=s,taxon=t)] = 1
        except:
            pass
#np.unique(SpeciesTaxonMAP, return_counts=True)


# In[9]:


# Species cutoff
sppcutoff = -1
# List of rarely caught species 
drop_spp = priorImportance.species.to_numpy()[priorImportance.max(["country"])<sppcutoff]


# In[10]:


# Number of species remaining with prior importance less than or equal to cutoff removed
#priorImportance.species.shape[0]-drop_spp.shape[0]


# In[11]:


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


# In[12]:


# Unique names of species in taxon list that are observed as taxon catches but have prior importance below cutoff
tmp_spp = np.unique(tmp_species_[tmp_species_spp_id[np.log1p(tmp["allCatch"])[tmp_tindx == 0]<sppcutoff]])
#len(tmp_spp)


# In[13]:


# Species in drop list that are actually observed as in taxon+species (taxon) list
tmp_spp = list(drop_spp[np.array([x in tmp_spp for x in drop_spp])])
#len(tmp_spp)


# In[14]:


#np.unique(priorImportance)


# In[15]:


# Grab landings data to see which taxons have catch
tmp_landings = tmp["allCatch"]
tmp_taxon = tmp["LandingsID"]
tmp_country = tmp["country"]
# Iterate over landings to ensure species are avaiable for taxon in country
for l,t,c in zip(tmp_landings,tmp_taxon,tmp_country):
    # Look for species landed with impossible priors
    try:
        if (priorImportance.sel(country=c,species=t)==-999)*(l>0):
            priorImportance.loc[dict(country=c,species=t)] = sppcutoff
            tmp_spp += [t]
            #print("Changed "+c+" "+t+' to '+str(sppcutoff) )
    # Look for taxon landed with no possible species 
    except:
        # Possible species for taxon
        tax_spp = (SpeciesTaxonMAP.sel(taxon=t).species[SpeciesTaxonMAP.sel(taxon=t)==1]).values
        # If all species are impossible for taxon
        if (priorImportance.sel(country=c,species=tax_spp).mean()==-999.)*(l>0):
            # Assign to possible species in nation
            if OutputSpeciesCountryMapFull.sel(country=c,species=tax_spp).sum()>0:
                axx = tax_spp[OutputSpeciesCountryMapFull.sel(country=c,species=tax_spp).to_numpy()>0]
                xflax = 'ok'
            else:
                # Assign to most likely spp given global max
                axx = tax_spp[(priorImportance.sel(species=tax_spp).max('country')==priorImportance.sel(species=tax_spp).max())]
                xflax = 'not present'
            # Change landings prior to cutoff value
            priorImportance.loc[dict(country=c,species=axx)] = sppcutoff
            tmp_spp += list(axx)
            #print("Impossible "+t+" changed "+c+", spp are "+xflax)
            #print(axx)
        
        # If all species for taxon are below data-reduction cutoff
        elif (priorImportance.sel(country=c,species=tax_spp).max()<sppcutoff)*(l>0):
            # Assign to possible species in nation
            if OutputSpeciesCountryMapFull.sel(country=c,species=tax_spp).sum()>0:
                axx = tax_spp[OutputSpeciesCountryMapFull.sel(country=c,species=tax_spp).to_numpy()>0]
                xflax = 'ok'
            else:
                # Assign to most likely spp given global max
                axx = tax_spp[(priorImportance.sel(species=tax_spp).max('country')==priorImportance.sel(species=tax_spp).max())]
                xflax = 'not present'
            # Change landings prior to cutoff value
            priorImportance.loc[dict(country=c,species=axx)] = sppcutoff
            tmp_spp += list(axx)
            #print("Low "+t+" changed "+c+", spp are "+xflax)
            #print(axx)


# In[16]:


# Create an empty DataFrame with the specified index and columns
scorez = list(np.unique(priorImportance).astype(int).astype(str))
tradeweightcounts = pd.DataFrame(index=priorImportance.country, columns=scorez).fillna(0)
for c in priorImportance.country.to_numpy():
    colz, valz = np.unique(priorImportance[priorImportance.country==c,:],return_counts=True)
    tradeweightcounts.loc[c,colz.astype(int).astype(str)] = valz
#tradeweightcounts


# In[17]:


# Add in species in field data
field_tmp = np.array(['Atlantoraja cyclhophora', 'Bathtoshia centroura',
       'Callorynchus callorynchus',
       'Dasyatis hypostigma', 'Fontitrygon geijskesi',
       'Hypanus bethalutzae', 'Mobula hypostoma', 'Narcine brasiliensis',
       'Pseudobatos horkelii', 'Pteroplatytrygon violacae',
       'Rhinoptera brasilisensis', 'Rioraja agassizi',
       'Scyliorhinus haekelii', 'Squalus albicaudus', 'Squatina occulta',
       'Zapteryx brevirostris','Apristurus brunneus', 'Bathyraja aleutica',
       'Bathyraja interrupta', 'Bathyraja murrayi',
       'Cephaloscyllium isabellum', 'Gollum attenuatus',
       'Oxynotus bruniensis', 'Rostroraja eglanteria'])
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
# Add in unreporting countries

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
meat_mask[meat_mask==0] = -9
meat_mask[meat_mask==1] = 0


# # Match FAO to observed data

# In[18]:


# Subset observed landings to match FAO countries
Obsspp = np.sort(np.array(list(set(ObsLandings_.species.values).intersection(priorImportance.species.values))))
Obscou = np.sort(np.array(list(set(ObsLandings_.country.values).intersection(priorImportance.country.values))))
ObsLandings = ObsLandings_.sel(country=Obscou, species=Obsspp)


# In[19]:


# Tmp empty data to match dimension of full species compliment for observed countries
tmpdata = priorImportance.copy()*-0
tmpdata.loc[dict(country=ObsLandings.country, species=ObsLandings.species)] = ObsLandings.values
ObsLandings = tmpdata.sel(country=Obscou)


# ## Set up split data

# In[20]:


# Calculate average total catch for target of softmax
TotalCatch = np.array([allCatch[CountryIDx==i].sum() for i in range(speciesCountryIDMap.shape[0])])/len(year_)
logTotalCatch = np.log(TotalCatch)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


#np.unique(taxon_[TaxonIDx[tindx == 0]][match(taxon_[TaxonIDx[tindx == 0]], list(species_))==None])


# In[22]:


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

# In[23]:


# Initial mask of species matched to possible taxon
InitTaxonMASK = speciesCountryIDMap.copy()
# Create empty mask to store observed taxa as possible
TaxonMASK = InitTaxonMASK*0


# In[24]:


# Reported taxa by country
Obs_tax_data = np.exp(txdata.drop(columns=['year','year_tax_id','country_tax_id','taxon_tax_id']
          ).groupby(['country','taxon']).mean()).rename(columns={"logReported_taxon_landings": "Reported_landings"})


# In[25]:


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


# # Update trade and landing weights

# In[26]:


# Grab trade weights
wdata = pd.read_csv(bd + "trade_weights.csv").copy()
# Empty array to hold values
tradeweights = pd.DataFrame(index=country_, columns=species_).fillna(0).to_numpy()
tradeweights.flags.writeable = True

# Make xarray and add importer dimension
tradeweights = (xr.DataArray(tradeweights, dims=('exporter', 'species'))
                .assign_coords({"exporter": country_,"species": species_})
                .expand_dims(importer=country_,axis=2)
               )
# Make a copy to make it writable
tradeweights = tradeweights.copy()


# In[27]:


# Fill in values
for index, row in wdata.iterrows():
    try:
        # Common export logOdds
        tradeweights.loc[dict(exporter=row.exporter,species=row.species)] = row.tradeweight
        # Domestic consumption logOdds
        tradeweights.loc[dict(exporter=row.exporter,species=row.species,importer=row.exporter)] = row.domesticweight
    except:
        pass


# In[28]:


# Make priors using relative odds to proportion total landings
SppPRIOR = priorImportance.to_numpy()


# In[29]:


# Create an empty DataFrame with the specified index and columns
scorez = list(np.unique(SppPRIOR).astype(int).astype(str))
tradeweightcounts = pd.DataFrame(index=country_, columns=scorez).fillna(0)

for c in country_:
    colz, valz = np.unique(SppPRIOR[country_==c,:],return_counts=True)
    tradeweightcounts.loc[c,colz.astype(int).astype(str)] = valz
#tradeweightcounts


# In[30]:


# Add updates to landing priors - NOV 2024
for index, row in wdata.iterrows():
    SppPRIOR[country_==row.exporter,species_==row.species] = row.landweight


# ## Re-weight scores

# In[31]:


# Downweight -2
poo = SppPRIOR.copy()
poo[poo==-2] = -3
# Calculate the expected proportions of species in each country
tmp = pd.DataFrame(np.exp(poo))
proportions_df = tmp.div(tmp.sum(axis=1), axis=0).fillna(0)
# Calculate log-odds for each proportion
# Adding a small constant to avoid log(0)
epsilon = 1e-6
SppPRIORadj = round(np.log(proportions_df / (1 - proportions_df + epsilon) + epsilon)+6).values

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


# In[32]:


tmp = np.array(['Carcharhinus acronotus', 'Carcharhinus brevipinna',
       'Dasyatis hypostigma', 'Heptranchias perlo',
       'Narcine brasiliensis', 'Rhinoptera bonasus',
       'Rhizoprionodon lalandii'])


# In[33]:


# Create an empty DataFrame with the specified index and columns
scorez = list(np.unique(SppPRIOR).astype(int).astype(str))
tradeweightcounts = pd.DataFrame(index=country_, columns=scorez).fillna(0)

for c in country_:
    colz, valz = np.unique(SppPRIOR[country_==c,:],return_counts=True)
    tradeweightcounts.loc[c,colz.astype(int).astype(str)] = valz
#tradeweightcounts


# ## Match landings and trade data

# In[34]:


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


# Import country keys
kdata = pd.read_csv(bd + "country_codes_V202409.csv")
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

# Drop all trade on code 30571
tdata = tdata[tdata["hscode_k"]!=30571]


# In[35]:


#list(filter(lambda x: 'fish' in x, ckey.description))


# In[36]:


# Make fresh/frozen fins dummy
tmp = np.array([0]*len(tdata['hs_description']))
tmp[tdata["hscode_k"]==30292] = 1
tmp[tdata["hscode_k"]==30392] = 1
tdata['fins'] = tmp

# Make rays dummy
tdata['rays'] = np.array(['rays' in l for l in tdata['hs_description']])*1
# Make 30488 (grouped code) sharks
maskx = tdata.hscode_k==30488
tdata.loc[maskx,'rays'] = 0

# Add shark or ray group
tmp = np.array(['sharks']*len(tdata['rays']))
tmp[tdata['rays']==1] = 'rays'
tdata['group'] = tmp


# # Weight for fins

# In[37]:


# Grab sharks in years that have fresh/frozen fins separated
tdata_17 = tdata.copy()[(tdata.year_t>=tdata[tdata.fins==1].year_t.min()) & (tdata.group=='sharks')]


# In[38]:


# Get total exports of fins and meat per year, exporter, importer combo
tmp = tdata_17.groupby(["ISOex_i","ISOim_j","year_t","fins"]).sum().reset_index()
# Grab 30292 and 30392 (fins) commodity
tmp_fins = tmp.copy()[tmp.fins==1]
# Sum over total per exporter, importer combo
tmp_sum = tmp[["ISOex_i","ISOim_j","quantity_q"]].groupby(["ISOex_i","ISOim_j"]).sum()
# Sum fins over total per year, exporter, importer
tmp_fins_sum = tmp_fins[["ISOex_i","ISOim_j","quantity_q"]].groupby(["ISOex_i","ISOim_j"]).sum()
# Grab proportion of fins where there are any
tmp_fin_props = (tmp_fins_sum/tmp_sum).reset_index()
# Make zero fins where there are none
maskx = tmp_fin_props.quantity_q.isna()
tmp_fin_props.loc[maskx,'quantity_q'] = 0
# Convert to proportions of meat
tmp_meat_props = tmp_fin_props.copy()
tmp_meat_props['prop_sharks'] = 1-tmp_meat_props.quantity_q
tmp_meat_props = tmp_meat_props.drop(columns=['quantity_q'])


# In[39]:


tdata_old = tdata.copy()
# Iterate over exporter/importer combos
for i in range(tmp_meat_props.shape[0]):
    exi = tmp_meat_props.ISOex_i[i]
    imi = tmp_meat_props.ISOim_j[i]
    shark_prop = tmp_meat_props.prop_sharks[i]
    # Re-scale observed sharks pre 2017 to remove fins fraction from fresh or frozen sharks
    maskx = ((tdata.year_t<2017) & (tdata.ISOex_i==exi) & (tdata.ISOim_j==imi)&(tdata.group=='sharks'))
    tdata.loc[maskx,'quantity_q'] = tdata.loc[maskx,'quantity_q']*shark_prop


# In[40]:


#plt.scatter(tdata_old.quantity_q,tdata.quantity_q)
#plt.xlabel('All shark trade')
#plt.ylabel('Adjusted shark trade')
#plt.title('Shark trade with fins removed 2012-2017');


# In[41]:


# Convert to live weights
tdata['estimated_live_weight'] = tdata.quantity_q
# Sharks conversion
tmpmask = tdata.group=='sharks'
tdata.loc[tmpmask,'estimated_live_weight'] = tdata.loc[tmpmask,'estimated_live_weight']*2
# Rays conversion
tmpmask = tdata.group=='rays'
tdata.loc[tmpmask,'estimated_live_weight'] = tdata.loc[tmpmask,'estimated_live_weight']*1.6


# In[42]:


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


# In[43]:


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

# Better country labels
biggest_countries_long = kdata.country_name_abbreviation[
    [list(kdata.iso_3digit_alpha).index(x) for x in biggest_countries]
].to_numpy()


# ## Deal with re-exports

# In[44]:


# # REEXPORTS - NEED TO CHECK THAT RAYS ARE COOL HERE TOO
#
# Currently removes trade that has no possible catch.
#
# NB:
#
# 1. Assumes catches in year t are traded in year t

# Pre-removals copy
tdata_copy = tdata.copy()
# Unique country list
tmp_c = np.unique(np.array(list(tdata.ISOex_i)+list(tdata.ISOim_j)))
# Unique commodity codes
tmp_u = tdata.group.unique()
# Landings per country per year per commodity
tmp_l = fdata.groupby(["country_code","year","group"]).sum().reset_index().sort_values("landed_weight", ascending=False)


# Empty list of identified re-exports
tmp = []

# ====================== Remove re-exports from trade =========================== #
# Iterate over years
for y in tdata.year_t.unique():
    # Grab values for year y 
    trad_ = tdata[tdata.year_t==y].groupby(["ISOex_i",'year_t','group']).sum().reset_index().sort_values("estimated_live_weight", ascending=False)
    # Grab total trade for year y
    trad = trad_.groupby(["ISOex_i",'year_t']).sum().reset_index().sort_values("estimated_live_weight", ascending=False)
    # Grab group trade for year y
    trad_s = trad_[trad_.group=='sharks']
    trad_r = trad_[trad_.group=='rays']
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
                maskx = (tdata.ISOex_i==e) & (tdata.year_t==y)
                tdata.loc[maskx,'estimated_live_weight'] = 0
                
            # If no catches (group+elasmos) to support shark trade, make trade zero to remove re-exports: 
            if sum(lx_s.landed_weight+lx_e.landed_weight)==0 and sum(tx_s.estimated_live_weight)>0:
                tmp += [sum(tx_s.estimated_live_weight)]
                maskx = (tdata.ISOex_i==e) & (tdata.year_t==y) & (tdata.group=='sharks')
                tdata.loc[maskx,'estimated_live_weight'] = 0
                
            # If no catches (group+elasmos) to support ray trade, make trade zero to remove re-exports:
            if sum(lx_r.landed_weight+lx_e.landed_weight)==0 and sum(tx_r.estimated_live_weight)>0:
                tmp += [sum(tx_r.estimated_live_weight)]
                maskx = (tdata.ISOex_i==e) & (tdata.year_t==y) & (tdata.group=='rays')
                tdata.loc[maskx,'estimated_live_weight'] = 0
                
            # If trade more than catches, make proportional within allowable group
            elif sum(lx.landed_weight)<sum(tx.estimated_live_weight):
                tmp += [sum(tx.estimated_live_weight)]
                # Grab proportion
                rrx = sum(lx.landed_weight)/sum(tx.estimated_live_weight)
                # Re-scale trade to proportion of total landings
                mask_s = (tdata.ISOex_i==e) & (tdata.year_t==y) & (tdata.group=='sharks')
                tdata.loc[mask_s,'estimated_live_weight'] = tdata.loc[mask_s,'estimated_live_weight']*rrx
                mask_r = (tdata.ISOex_i==e) & (tdata.year_t==y) & (tdata.group=='rays')
                tdata.loc[mask_r,'estimated_live_weight'] = tdata.loc[mask_r,'estimated_live_weight']*rrx
        else:
            mask = ((tdata.ISOex_i==e) & (tdata.year_t==y))
            tdata.loc[mask,'estimated_live_weight'] = 0
            #print(e,y)        

# ## Restrict landings, seafood trade and meat trade to biggest countries

# Original
tmp_x = (tdata_copy.estimated_live_weight)
# Updated
tmp_y = (tdata.estimated_live_weight)
# Countries with trade reduced
iredux = (tdata_copy.estimated_live_weight-tdata.estimated_live_weight)>1
# Grap re-exports data
ReExports = pd.DataFrame(zip(tmp_x[iredux],tmp_y[iredux],tdata.ISOex_i[iredux].to_numpy(),tdata_copy.year_t[iredux],tdata_copy.group[iredux]),columns=['Original','Reduced','exporter','year','group'])
ReExports['Net_diff'] = ReExports.Original-ReExports.Reduced
ReExports['Exporter'] = kdata.country_name_abbreviation[[list(kdata.iso_3digit_alpha).index(x) for x in ReExports.exporter]].to_numpy()

# Table of re-exporting countries
tmp = ReExports.groupby(['Exporter']).sum().sort_values(by='Net_diff',ascending=False).drop(columns='exporter')
tmp.to_csv('ReExport_totals_AUGM.csv')
# Local augmented re-exports
ReExports_AUGM = ReExports

# = = = = = = = = = = = = = #
# Remove zeros from trade
tdata = tdata[tdata.estimated_live_weight!=0]

# = = = = = = = = = = = = = #
# Remove fins
tdata = tdata[tdata['product']=='meat']


# # Check FAO trade data

# In[45]:


# FAO trade data
rdata = pd.read_csv(bd + "FAO_elasmo.csv")
# Drop rows with zero trade
rdata = rdata[rdata.trade_quantity>0]
# Drop rows outside of 2012-2019
rdata = rdata[rdata.year<=2019]
# Drop NA
rdata = rdata[rdata.trade_quantity.notna()]
rdata = rdata[rdata.commodity_fao.notna()]

# Drop other trade
rdata = rdata[rdata.commodity_fao!='Shark fins, smoked, dried, whether or not salted, etc.']
rdata = rdata[rdata.commodity_fao!='Shark fins, prepared or preserved']
rdata = rdata[rdata.commodity_fao!='Shark fins, frozen']
rdata = rdata[rdata.commodity_fao!='Shark fins, salted and in brine but not dried or smoked']
rdata = rdata[rdata.commodity_fao!='Shark fins, dried, unsalted']
rdata = rdata[rdata.commodity_fao!='Sharks, dried, whether or not salted, but not smoked']
rdata = rdata[rdata.commodity_fao!='Shark oil']
rdata = rdata[rdata.commodity_fao!='Shark liver oil']
rdata = rdata[rdata.commodity_fao!='Sharks, fillets, dried, salted or in brine']

# Drop countries outside of top traders
tindx = (match(biggest_countries_long,list(rdata.reporting_country.values))[
         match(biggest_countries_long,list(rdata.reporting_country.values))!=None]
        )
rdata = rdata.iloc[tindx]


# In[46]:


# Grab codes
FAOcodes = pd.DataFrame({'code':rdata.commodity_fao.unique()})
FAOcodes['group'] = np.array(['rays','sharks'])[FAOcodes.code.str.contains('shark').values*1]
FAOcodes.group[(FAOcodes.code.str.contains('shark',case=False, regex=False, na=False).values*1
                +FAOcodes.code.str.contains('ray',case=False, regex=False, na=False).values*1)==2] = 'both'


# In[47]:


# Write external key
pd.DataFrame(columns=FAOcodes.code,index=srkey.species_binomial).to_csv('FAO_elasmo_key.csv')


# In[48]:


rdata[rdata.trade_quantity>50]


# ## Setup landings data

# In[49]:


# Split out shark and ray trade
tdata_cut = tdata[
    ((tdata.ISOex_i).isin(biggest_countries))
    & ((tdata.ISOim_j).isin(biggest_countries))
]
# dropping missing values
tdata_cut = tdata_cut.drop("Unnamed: 0", axis="columns").dropna().reset_index(drop=True)
# Summarize data by importer export commodity
tdata_cut = tdata_cut.groupby(["ISOex_i", "ISOim_j", "group",'year_t']).sum().reset_index()


# In[50]:


# Get average per year
tdata_cut = (tdata_cut[["ISOex_i", "ISOim_j", "group","year_t",'quantity_q','estimated_live_weight']]
             .groupby(["ISOex_i", "ISOim_j", "group"])
             .mean()
             .drop(columns=['year_t'])
             .reset_index()
            )


# In[51]:


# we just care about whether it's a shark or ray, not if it's also fresh or frozen
tdata_cut = tdata_cut.sort_values(["ISOex_i", "ISOim_j", "group"])

# Shark data
tdata_cut_sharks = (
    tdata_cut[tdata_cut.group == "sharks"]
    .groupby(["ISOex_i", "ISOim_j"])
    .sum()
    .reset_index()
)

# Ray data
tdata_cut_rays = (
    tdata_cut[tdata_cut.group == "rays"]
    .groupby(["ISOex_i", "ISOim_j"])
    .sum()
    .reset_index()
)


# In[ ]:





# In[52]:


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
                pd.DataFrame({'exporter': [0], 'importer': [0]}, index=[i]),
                unreliability_score,
            ]
        )

unreliability_score = unreliability_score.sort_index().fillna(tmp_maxval)
unreliability_score = unreliability_score.reindex(
    sorted(unreliability_score.columns), axis=1
)


# ## Setup taxon masking

# In[53]:


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

# In[54]:


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
trade_mask[trade_mask==0] = -9
trade_mask[trade_mask>0] = 0
# Remove species not used for meat
#trade_mask = trade_mask+meat_mask[None,:,None]
trade_mask[trade_mask<0] = -9

# Mask for trade softmax to zero out species with all -9
NoSPP_Mask = (((trade_mask==-9).sum(2)!=len(country_))*1)

# Mask for blue shark relative odds importer preferences
BSmask = np.zeros(shape=trade_mask[0].shape)
BSmask[list(species_).index('Prionace glauca')] = -9

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


# In[55]:


# Observed data country index 
obs_exporter_idx = np.array([country_to_idx_map[x] for x in Obscou])

# Tmp fix of priors set too low for observed species
tmp = ((SppPRIOR[obs_exporter_idx,:]==-999) & (ObsLandings.values>0))

for t in range(len(tmp.sum(1))):
    if tmp[t].sum()>0:
        tmpc = ObsLandings.country[t].values
        tmps = ObsLandings.species[tmp[t]].values
        tmpo = ObsLandings.sel(country=tmpc,species=tmps).values
        tmpm = priorImportance.sel(country=tmpc,species=tmps).values
        #print(tmpc,tmps,tmpo,tmpm)
        # tmp fix of priors
        SppPRIOR[country_==tmpc,np.isin(species_,tmps)] = 1
        SppPRIORadj[country_==tmpc,np.isin(species_,tmps)] = 1


# ## Trade weights

# In[56]:


# Replace values in revision tradeweights with tradeIportance priors
for e in country_:
    for s in species_:
        tradeweights.values[country_==c,species_==s,country_!=c] = tradeImportance.sel(country=e,species=s).values


# In[57]:


# Replace -999 with -9
tradeweights.values[tradeweights==-999] = -9


# In[58]:


# Grab bilateral trade weights
bdata = pd.read_csv(bd + "bilateral_trade.csv").copy()

# Update specific weights
for r in bdata.itertuples():
    tradeweights.values[country_==r.exporter,species_==r.species,country_==r.importer] = r.tradeweight


# In[59]:


COORDS = {
    "exporter": biggest_countries,
    "importer": biggest_countries,
    "obs_exporter": country_[obs_exporter_idx],
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

