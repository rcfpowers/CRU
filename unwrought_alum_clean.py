import pandas as pd
import re
import numpy as np


def get_majority_owner(ownership_str):
    if not isinstance(ownership_str, str):
        return None

    pattern = r'(.*?)-\s*([\d.]+)%'
    matches = re.findall(pattern, ownership_str)

    max_owner = None
    max_percent = -1

    for name, percent in matches:
        name_clean = name.strip().strip(', ')
        if re.search(r'\b(other|various)\b', name_clean, flags=re.IGNORECASE):
            continue
        try:
            percent = float(percent)
            if percent > max_percent:
                max_percent = percent
                max_owner = name_clean
        except ValueError:
            continue

    return max_owner


def set_var_aggregation(df, var_str_list, agg_level, agg_str, product_filter=None):
    """Aggregate specified variables into a single new variable"""
    df_to_agg = df[df['Variable'].isin(var_str_list)].copy()

    if product_filter is not None:
        df_to_agg = df_to_agg[df_to_agg['Product'] == product_filter]

    if df_to_agg.empty:
        return df

    df_to_agg['Value'] = df_to_agg['Value'].fillna(0)

    agg_dict = {'Value': 'sum'}
    other_cols = [col for col in df_to_agg.columns
                  if col not in agg_level and col not in ['Variable', 'Value']]

    for col in other_cols:
        agg_dict[col] = 'first'

    df_group = df_to_agg.groupby(agg_level, as_index=False).agg(agg_dict)
    df_group['Variable'] = agg_str

    if product_filter is not None:
        df_remaining = df[~((df['Variable'].isin(var_str_list)) &
                          (df['Product'] == product_filter))]
    else:
        df_remaining = df[~df['Variable'].isin(var_str_list)]

    df_result = pd.concat([df_remaining, df_group], ignore_index=True)
    return df_result


df = pd.read_csv("Commodities/unwrought_aluminum.csv")

is_prod = df['Variable'] == 'Production, kt'
keys = pd.MultiIndex.from_frame(df[['Asset', 'Product']])
has_casthouse = pd.MultiIndex.from_frame(
    df.loc[is_prod & (df['Category'] == 'Casthouse'), ['Asset', 'Product']])
drop = is_prod & (df['Category'] == 'Production') & keys.isin(has_casthouse)
df = df[~drop]

df.drop(columns=['Asset - Area', 'Asset - Basin name', 'Asset - Coalfield name',
                 'Asset - Mine type', 'Asset ID (select all)', 'Access codes (select all)',
                 'Scale', 'Asset - Service', 'Asset - Ore mineralisation', 'Category',
                 'Asset - Technology'], inplace=True)

df = df.melt(id_vars=['Commodity', 'Asset', 'Asset - Owners', 'Asset - Operator',
                      'Asset - Region', 'Asset - State',
                      'Variable', 'Product', 'Operating status', 'Country', 'Units'],
             var_name="Year", value_name="Value")

df = set_var_aggregation(df, ['Soderberg, Scope 1 & 2, t CO2e/t', 'Prebake, Scope 1 & 2, t CO2e/t'],
                         ['Asset', 'Year'], 'Upstream, Scope 1 & 2, t CO2e/t')
df = set_var_aggregation(df, ['Soderberg, Scope 1, kt CO2e', 'Prebake, Scope 1, kt CO2e'],
                         ['Asset', 'Year'], 'Upstream, Scope 1, kt CO2e')
df = set_var_aggregation(df, ['Soderberg, Scope 2, kt CO2e', 'Prebake, Scope 2, kt CO2e'],
                         ['Asset', 'Year'], 'Upstream, Scope 2, kt CO2e')
df = set_var_aggregation(df, ['Soderberg, Scope 1 & 2 & 3, t CO2e/t',
                              'Prebake, Scope 1 & 2 & 3, t CO2e/t'],
                         ['Asset', 'Year'], 'Upstream, Scope 1 & 2 & 3, t CO2e/t')
df = set_var_aggregation(df, ["Potline power emissions, Scope 2, t CO2e/t",
                              "Potline power emissions, Scope 1, t CO2e/t"],
                         ['Asset', 'Year'], 'Upstream Power, Scope 1 & 2, t CO2e/t')

df = set_var_aggregation(df, ['Potroom power consumption, MWh/t',
                              'Ancillary power consumption, MWh/t'],
                         ['Asset', 'Product', 'Year'], 'Upstream Power, MWh/t rolled product')

df = set_var_aggregation(df, ["Fuel, continuous ingot casters, TJ",
                              "Fuel, batch ingot/sow casters, TJ"],
                         ['Asset', 'Product', 'Year'], 'Fuel, TJ',
                         product_filter='Primary ingot')

df = set_var_aggregation(df, ["Fuel, continuous alloy casters, TJ",
                              "Fuel, batch alloy casters, TJ"],
                         ['Asset', 'Product', 'Year'], 'Fuel, TJ',
                         product_filter='Foundry alloy')

df = set_var_aggregation(df, ["Postconsumer scrap, %", "Preconsumer scrap, %"],
                         ['Asset', 'Year'], 'Scrap, %', product_filter='Aluminium')

mask = df['Variable'] == 'Power, continuous alloy casters, kWh'
df.loc[mask, 'Units'] = 'MWh'
df.loc[mask, 'Variable'] = 'Power, continuous alloy casters, MWh'

mask = df['Variable'] == 'Total potroom power consumed, GWh'
df.loc[mask, 'Value'] = df.loc[mask, 'Value'] * 1000
df.loc[mask, 'Units'] = 'MWh'
df.loc[mask, 'Variable'] = 'Total potroom power consumed, MWh'

mask = df['Variable'] == 'Fuel, TJ'
df.loc[mask, 'Value'] = df.loc[mask, 'Value'] * 1000
df.loc[mask, 'Units'] = 'GJ'
df.loc[mask, 'Variable'] = 'Fuel, GJ'

mask = df['Variable'] == 'Fuel, casthouse, TJ'
df.loc[mask, 'Value'] = df.loc[mask, 'Value'] * 1000
df.loc[mask, 'Units'] = 'GJ'
df.loc[mask, 'Variable'] = 'Fuel, casthouse, GJ'

mask = df['Variable'] == 'Smelter fuel required, TJ'
df.loc[mask, 'Value'] = df.loc[mask, 'Value'] * 1000
df.loc[mask, 'Units'] = 'GJ'
df.loc[mask, 'Variable'] = 'Fuel, smelter, GJ'

power_rename_map = {
    'Power, continuous ingot casters, MWh': 'Power, ingot, MWh',
    'Power, batch ingot/sow casters, MWh': 'Power, ingot, MWh',
    'Power, continuous alloy casters, MWh': 'Power, alloy, MWh',
    'Power, batch alloy casters, MWh': 'Power, alloy, MWh',
}
df['Variable'] = df['Variable'].map(lambda x: power_rename_map.get(x, x))

df = set_var_aggregation(df, ['Power, ingot, MWh'],
                         ['Asset', 'Product', 'Year'], "Power, caster, MWh",
                         product_filter='Primary ingot')

df = set_var_aggregation(df, ['Power, alloy, MWh'],
                         ['Asset', 'Product', 'Year'], "Power, caster, MWh",
                         product_filter='Foundry alloy')

df = set_var_aggregation(df, ['Power, MWh/t CC rod', 'Power, MWh/t CC strip',
                              'Power, MWh/t extrusion billet',
                              'Power, MWh/t rolling slab'],
                         ['Asset', 'Product', 'Year'], "Power_P, MWh/t product")

df = set_var_aggregation(df, ['Fuel, GJ/t CC rod', "Fuel, GJ/t CC strip",
                              'Fuel, GJ/t extrusion billet', "Fuel, GJ/t rolling slab"],
                         ['Asset', 'Product', 'Year'], "Fuel_P, GJ/t product")

key_variables = [
    'Fuel, GJ', 'Fuel, casthouse, GJ', "Power_P, MWh/t product", "Fuel_P, GJ/t product",
    'Casthouse emissions, Scope 1, kt CO2e', "Production, kt",
    "Power, MWh", "Power, caster, MWh", 'Upstream Power, MWh/t rolled product',
    "Casthouse production, kt", "Casthouse emissions, Scope 2, kt CO2e",
    "Power, casthouse, MWh", 'Scrap, %', "Upstream, Scope 1 & 2, t CO2e/t",
    "Upstream, Scope 2, kt CO2e", "Upstream, Scope 1, kt CO2e",
    'Total CO2e emissions costs, $/t', 'Site costs, $/t', 'Business costs, $/t',
    "Primary metal, kt", "Capacity, kt", 'Primary aluminium production, kt',
    'Upstream, Scope 1 & 2 & 3, t CO2e/t', "Casthouse, Scope 1 & 2, t CO2e/t",
    'Upstream Power, Scope 1 & 2, t CO2e/t', "Potline power emissions, Scope 2, t CO2e/t",
    "Potline power emissions, Scope 1, t CO2e/t", 'Power emissions, Scope 1 & 2, t CO2e',
    'Total potroom power consumed, MWh', 'Ancillary emissions, Scope 1 & 2, t CO2e/t',
    "Aggregate/other renewables power, MWh", "Thermal coal power, MWh", "Self-generation share, %",
    "Natural gas power, MWh", "Hydro power, MWh", "Nuclear power, MWh", 'Total employees, #',
    'Fuel, smelter, GJ', 'Total site, Scope 1, kt CO2e', 'Total site, Scope 2, kt CO2e'
]

df = df[df['Variable'].isin(key_variables)]

df['Variable'] = df['Variable'].str.rsplit(',', n=1).str[0].str.strip()

rename_dict = {
    'Fuel, casthouse': 'Fuel Casthouse',
    'Casthouse emissions, Scope 1': 'Scope 1 Casthouse',
    'Casthouse emissions, Scope 2': 'Scope 2 Casthouse',
    'Power, casthouse': 'Power Casthouse',
    'Power, caster': 'Power Caster',
    'Upstream Power': 'Power Upstream',
    'Upstream, Scope 1 & 2': 'Scope 1 & 2 Upstream',
    'Upstream, Scope 1': 'Scope 1 Upstream',
    'Upstream, Scope 2': 'Scope 2 Upstream',
    'Primary aluminium production': 'Primary Production',
    'Upstream, Scope 1 & 2 & 3': 'Scope 1 & 2 & 3 Upstream',
    'Casthouse, Scope 1 & 2': 'Scope 1 & 2 Casthouse Intensity',
    'Upstream Power, Scope 1 & 2': 'Scope 1 & 2 Upstream Power',
    'Ancillary emissions, Scope 1 & 2': 'Scope 1 & 2 Ancillary',
    'Power emissions, Scope 1 & 2': 'Scope 1 & 2 Power Emit',
    "Aggregate/other renewables power": "Renewable Power",
    "Thermal coal power": "Coal Power",
    "Self-generation share": "Self Power",
    "Natural gas power": 'Natural Gas Power',
    "Hydro power": "Hydro Power",
    "Nuclear power": "Nuclear Power",
    'Total potroom power consumed': 'Potroom Power',
    'Total employees': 'employees_tot',
    'Fuel, smelter': 'smelt fuel',
    'Total site, Scope 1': 'total site scope 1',
    'Total site, Scope 2': 'total site scope 2'
}

for old, new in rename_dict.items():
    df.loc[df['Variable'] == old, 'Variable'] = new

df = df[df['Operating status'] == 'Operating']
df = df[pd.to_numeric(df['Year'], errors='coerce') <= 2024].copy()

df['Owners'] = df['Asset - Owners'].apply(get_majority_owner)
df.drop(columns=['Asset - Owners'], inplace=True)

df['Value'] = pd.to_numeric(df['Value'], errors='coerce').fillna(0)
df['Units'] = df['Units'].fillna('Unknown')

all_asset_years = df[['Asset', 'Year']].drop_duplicates()

# Scope 1 Casthouse
s1_data = df[df['Variable'] == 'Scope 1 Casthouse'][['Asset', 'Year', 'Value']].copy()
s1_data = s1_data.rename(columns={'Value': 'Casting_1'})
s1_data = all_asset_years.merge(s1_data, on=['Asset', 'Year'], how='left')
s1_data['Casting_1'] = s1_data['Casting_1'].fillna(0)

# Fuel Casthouse
fuel_data = df[df['Variable'] == 'Fuel Casthouse'][['Asset', 'Year', 'Value']].copy()
fuel_data = fuel_data.rename(columns={'Value': 'Casting_F'})
fuel_data = all_asset_years.merge(fuel_data, on=['Asset', 'Year'], how='left')
fuel_data['Casting_F'] = fuel_data['Casting_F'].fillna(0)

# Scope 2 Casthouse
scope2_data = df[df['Variable'] == 'Scope 2 Casthouse'][['Asset', 'Year', 'Value']].copy()
scope2_data = all_asset_years.merge(scope2_data, on=['Asset', 'Year'], how='left')
scope2_data['Value'] = scope2_data['Value'].fillna(0)
scope2_data = scope2_data.rename(columns={'Value': 'Casting_2'})

# Scope 1 & 2 Casthouse Intensity
c12_d = df[df['Variable'] == 'Scope 1 & 2 Casthouse Intensity'][['Asset', 'Year', 'Value']].copy()
c12_d = all_asset_years.merge(c12_d, on=['Asset', 'Year'], how='left')
c12_d['Value'] = c12_d['Value'].fillna(0)
c12_d = c12_d.rename(columns={'Value': 'Casting_1_2_I'})

# Power Casthouse
power_data = df[df['Variable'] == 'Power Casthouse'][['Asset', 'Year', 'Value']].copy()
power_data = power_data.rename(columns={'Value': 'Casting_P'})
power_data = all_asset_years.merge(power_data, on=['Asset', 'Year'], how='left')
power_data['Casting_P'] = power_data['Casting_P'].fillna(0)

# Power Potroom
potpwr_data = df[df['Variable'] == 'Potroom Power'][['Asset', 'Year', 'Value']].copy()
potpwr_data = potpwr_data.rename(columns={'Value': 'Potroom_P'})
potpwr_data = all_asset_years.merge(potpwr_data, on=['Asset', 'Year'], how='left')
potpwr_data['Potroom_P'] = potpwr_data['Potroom_P'].fillna(0)

# Upstream Scope 1 & 2
upstream_data = df[df['Variable'] == 'Scope 1 & 2 Upstream'][['Asset', 'Year', 'Value']].copy()
upstream_data = upstream_data.rename(columns={'Value': 'Firm_Ups_1_2_I'})
upstream_data = all_asset_years.merge(upstream_data, on=['Asset', 'Year'], how='left')
upstream_data['Firm_Ups_1_2_I'] = upstream_data['Firm_Ups_1_2_I'].fillna(0)

# Upstream Scope 1
upstream_s1_data = df[df['Variable'] == 'Scope 1 Upstream'][['Asset', 'Year', 'Value']].copy()
upstream_s1_data = upstream_s1_data.rename(columns={'Value': 'Firm_Ups_1'})
upstream_s1_data = all_asset_years.merge(upstream_s1_data, on=['Asset', 'Year'], how='left')
upstream_s1_data['Firm_Ups_1'] = upstream_s1_data['Firm_Ups_1'].fillna(0)

# Upstream Scope 2
upstream_s2_data = df[df['Variable'] == 'Scope 2 Upstream'][['Asset', 'Year', 'Value']].copy()
upstream_s2_data = upstream_s2_data.rename(columns={'Value': 'Firm_Ups_2'})
upstream_s2_data = all_asset_years.merge(upstream_s2_data, on=['Asset', 'Year'], how='left')
upstream_s2_data['Firm_Ups_2'] = upstream_s2_data['Firm_Ups_2'].fillna(0)

# Upstream Scope 1 & 2 & 3
upstream3_data = df[df['Variable'] == 'Scope 1 & 2 & 3 Upstream'][['Asset', 'Year', 'Value']].copy()
upstream3_data = all_asset_years.merge(upstream3_data, on=['Asset', 'Year'], how='left')
upstream3_data['Value'] = upstream3_data['Value'].fillna(0)
upstream3_data = upstream3_data.rename(columns={'Value': 'Firm_Ups_1_2_3_I'})

# Power Emissions
power_emit_data = df[df['Variable'] == 'Scope 1 & 2 Power Emit'][['Asset', 'Year', 'Value']].copy()
power_emit_data = all_asset_years.merge(power_emit_data, on=['Asset', 'Year'], how='left')
power_emit_data['Value'] = power_emit_data['Value'].fillna(0)
power_emit_data = power_emit_data.rename(columns={'Value': 'Firm_1_2_P_I'})

# Upstream Power
up_p_data = df[df['Variable'] == 'Scope 1 & 2 Upstream Power'][['Asset', 'Year', 'Value']].copy()
up_p_data = all_asset_years.merge(up_p_data, on=['Asset', 'Year'], how='left')
up_p_data['Value'] = up_p_data['Value'].fillna(0)
up_p_data = up_p_data.rename(columns={'Value': 'Ups_P_1_2'})

# Upstream Power
up_power_mwh_data = df[df['Variable'] == 'Power Upstream'][['Asset', 'Year', 'Value']].copy()
up_power_mwh_data = all_asset_years.merge(up_power_mwh_data, on=['Asset', 'Year'], how='left')
up_power_mwh_data['Value'] = up_power_mwh_data['Value'].fillna(0)
up_power_mwh_data = up_power_mwh_data.rename(columns={'Value': 'Ups_P'})

# Scrap
scrap_data = df[df['Variable'] == 'Scrap'][['Asset', 'Year', 'Value']].copy()
scrap_data = all_asset_years.merge(scrap_data, on=['Asset', 'Year'], how='left')
scrap_data['Value'] = scrap_data['Value'].fillna(0)
scrap_data = scrap_data.rename(columns={'Value': 'Scrap'})

# Renewable Power
renew_data = df[df['Variable'] == 'Renewable Power'][['Asset', 'Year', 'Value']].copy()
renew_data = all_asset_years.merge(renew_data, on=['Asset', 'Year'], how='left')
renew_data['Value'] = renew_data['Value'].fillna(0)
renew_data = renew_data.rename(columns={'Value': 'Renewable_Power'})

# Coal Power
coal_data = df[df['Variable'] == 'Coal Power'][['Asset', 'Year', 'Value']].copy()
coal_data = all_asset_years.merge(coal_data, on=['Asset', 'Year'], how='left')
coal_data['Value'] = coal_data['Value'].fillna(0)
coal_data = coal_data.rename(columns={'Value': 'Coal_Power'})

# Self Power
self_data = df[df['Variable'] == 'Self Power'][['Asset', 'Year', 'Value']].copy()
self_data = all_asset_years.merge(self_data, on=['Asset', 'Year'], how='left')
self_data['Value'] = self_data['Value'].fillna(0)
self_data = self_data.rename(columns={'Value': 'Self_Power'})

# Hydro Power
hydro_data = df[df['Variable'] == 'Hydro Power'][['Asset', 'Year', 'Value']].copy()
hydro_data = all_asset_years.merge(hydro_data, on=['Asset', 'Year'], how='left')
hydro_data['Value'] = hydro_data['Value'].fillna(0)
hydro_data = hydro_data.rename(columns={'Value': 'Hydro_Power'})

# Nuclear Power
nuc_data = df[df['Variable'] == 'Nuclear Power'][['Asset', 'Year', 'Value']].copy()
nuc_data = all_asset_years.merge(nuc_data, on=['Asset', 'Year'], how='left')
nuc_data['Value'] = nuc_data['Value'].fillna(0)
nuc_data = nuc_data.rename(columns={'Value': 'Nuclear_Power'})

# Natural Gas Power
nat_data = df[df['Variable'] == 'Natural Gas Power'][['Asset', 'Year', 'Value']].copy()
nat_data = all_asset_years.merge(nat_data, on=['Asset', 'Year'], how='left')
nat_data['Value'] = nat_data['Value'].fillna(0)
nat_data = nat_data.rename(columns={'Value': 'Natural_Gas_Power'})

# Ancillary
ancillary_data = df[df['Variable'] == 'Scope 1 & 2 Ancillary'][['Asset', 'Year', 'Value']].copy()
ancillary_data = all_asset_years.merge(ancillary_data, on=['Asset', 'Year'], how='left')
ancillary_data['Value'] = ancillary_data['Value'].fillna(0)
ancillary_data = ancillary_data.rename(columns={'Value': 'Firm_1_2_Ancillary'})

# Casthouse Production
prod_data = df[df['Variable'] == 'Casthouse production'][['Asset', 'Year', 'Value']].copy()
prod_data = prod_data.rename(columns={'Value': 'Casting_Prod'})
prod_data = all_asset_years.merge(prod_data, on=['Asset', 'Year'], how='left')
prod_data['Casting_Prod'] = prod_data['Casting_Prod'].fillna(0)

# Primary Production
prim_data = df[df['Variable'] == 'Primary Production'][['Asset', 'Year', 'Value']].copy()
prim_data = prim_data.rename(columns={'Value': 'Primary_Prod'})
prim_data = all_asset_years.merge(prim_data, on=['Asset', 'Year'], how='left')
prim_data['Primary_Prod'] = prim_data['Primary_Prod'].fillna(0)

# Employee
empl_data = df[df['Variable'] == 'employees_tot'][['Asset', 'Year', 'Value']].copy()
empl_data = empl_data.rename(columns={'Value': 'employees'})
empl_data = all_asset_years.merge(empl_data, on=['Asset', 'Year'], how='left')
empl_data['employees'] = empl_data['employees'].fillna(0)

# Smelt
smelt_data = df[df['Variable'] == 'smelt fuel'][['Asset', 'Year', 'Value']].copy()
smelt_data = smelt_data.rename(columns={'Value': 'smelt_fuel'})
smelt_data = all_asset_years.merge(smelt_data, on=['Asset', 'Year'], how='left')
smelt_data['smelt_fuel'] = smelt_data['smelt_fuel'].fillna(0)

# Total Site 1
total_site_1 = df[df['Variable'] == 'total site scope 1'][['Asset', 'Year', 'Value']].copy()
total_site_1 = total_site_1.rename(columns={'Value': 'total_site_1'})
total_site_1 = all_asset_years.merge(total_site_1, on=['Asset', 'Year'], how='left')
total_site_1['total_site_1'] = total_site_1['total_site_1'].fillna(0)

# Total Site 2
total_site_2 = df[df['Variable'] == 'total site scope 2'][['Asset', 'Year', 'Value']].copy()
total_site_2 = total_site_2.rename(columns={'Value': 'total_site_2'})
total_site_2 = all_asset_years.merge(total_site_2, on=['Asset', 'Year'], how='left')
total_site_2['total_site_2'] = total_site_2['total_site_2'].fillna(0)

e_cost_data = df[df['Variable'] == 'Total CO2e emissions costs'][['Asset', 'Year', 'Value']].copy()
e_cost_data = e_cost_data.rename(columns={'Value': 'Carbon_Cost'})
e_cost_data = e_cost_data.groupby(['Asset', 'Year'], as_index=False)['Carbon_Cost'].first()
e_cost_data = all_asset_years.merge(e_cost_data, on=['Asset', 'Year'], how='left')
e_cost_data['Carbon_Cost'] = e_cost_data['Carbon_Cost'].fillna(0)

site_cost_data = df[df['Variable'] == 'Site costs'][['Asset', 'Year', 'Value']].copy()
site_cost_data = site_cost_data.rename(columns={'Value': 'Site_Cost'})
site_cost_data = site_cost_data.groupby(['Asset', 'Year'], as_index=False)['Site_Cost'].first()
site_cost_data = all_asset_years.merge(site_cost_data, on=['Asset', 'Year'], how='left')
site_cost_data['Site_Cost'] = site_cost_data['Site_Cost'].fillna(0)

df = df.merge(s1_data, on=['Asset', 'Year'], how='left')
df = df.merge(fuel_data, on=['Asset', 'Year'], how='left')
df = df.merge(scope2_data, on=['Asset', 'Year'], how='left')
df = df.merge(power_data, on=['Asset', 'Year'], how='left')
df = df.merge(power_emit_data, on=['Asset', 'Year'], how='left')
df = df.merge(upstream_data, on=['Asset', 'Year'], how='left')
df = df.merge(upstream_s1_data, on=['Asset', 'Year'], how='left')
df = df.merge(upstream_s2_data, on=['Asset', 'Year'], how='left')
df = df.merge(ancillary_data, on=['Asset', 'Year'], how='left')
df = df.merge(upstream3_data, on=['Asset', 'Year'], how='left')
df = df.merge(up_p_data, on=['Asset', 'Year'], how='left')
df = df.merge(up_power_mwh_data, on=['Asset', 'Year'], how='left')
df = df.merge(prod_data, on=['Asset', 'Year'], how='left')
df = df.merge(prim_data, on=['Asset', 'Year'], how='left')
df = df.merge(scrap_data, on=['Asset', 'Year'], how='left')
df = df.merge(c12_d, on=['Asset', 'Year'], how='left')
df = df.merge(e_cost_data, on=['Asset', 'Year'], how='left')
df = df.merge(site_cost_data, on=['Asset', 'Year'], how='left')
df = df.merge(nat_data, on=['Asset', 'Year'], how='left')
df = df.merge(self_data, on=['Asset', 'Year'], how='left')
df = df.merge(coal_data, on=['Asset', 'Year'], how='left')
df = df.merge(renew_data, on=['Asset', 'Year'], how='left')
df = df.merge(hydro_data, on=['Asset', 'Year'], how='left')
df = df.merge(nuc_data, on=['Asset', 'Year'], how='left')
df = df.merge(potpwr_data, on=['Asset', 'Year'], how='left')
df = df.merge(empl_data, on=['Asset', 'Year'], how='left')
df = df.merge(smelt_data, on=['Asset', 'Year'], how='left')
df = df.merge(total_site_1, on=['Asset', 'Year'], how='left')
df = df.merge(total_site_2, on=['Asset', 'Year'], how='left')

variables_to_remove = [
    'Scope 1 Casthouse', 'Fuel Casthouse', 'Scope 2 Casthouse', 'Scrap',
    'Power Casthouse', 'Scope 1 & 2 Upstream', 'Scope 1 Upstream', 'Scope 2 Upstream',
    'Casthouse production', 'Primary production', 'Scope 1 & 2 & 3 Upstream',
    'Scope 1 & 2 Casthouse Intensity', 'total site scope 2', 'total site scope 1',
    'Scope 1 & 2 & 3 Casthouse Intensity', 'Scope 1 & 2 Upstream Power', 'Power Upstream',
    'Scope 1 & 2 Ancillary', 'Scope 1 & 2 Power Emit', "Renewable Power", 'employees_tot',
    "Coal Power", "Self Power", 'Natural Gas Power', 'Hydro Power', "Nuclear Power",
    'Potroom Power', 'smelt fuel'
]

df = df[~df['Variable'].isin(variables_to_remove)]

df_p = df.copy()
df_p['Variable'] = df_p['Variable'].str.rsplit(',', n=1).str[0].str.strip()

df_p_agg = df_p.groupby(['Asset', 'Year', 'Product', 'Variable'], as_index=False).agg({
    'Value': 'sum',
    'Units': 'first',
    'Commodity': 'first',
    'Owners': 'first',
    'Asset - Operator': 'first',
    'Operating status': 'first',
    'Country': 'first',
    'Asset - Region': 'first',
    'Asset - State': 'first',
    'Casting_1': 'first',
    'Casting_F': 'first',
    'Casting_2': 'first',
    'Casting_P': 'first',
    'Potroom_P': 'first',
    'Firm_Ups_1_2_I': 'first',
    'Firm_Ups_1': 'first',
    'Firm_Ups_2': 'first',
    'Casting_Prod': 'first',
    'Primary_Prod': 'first',
    'Firm_Ups_1_2_3_I': 'first',
    'Casting_1_2_I': 'first',
    'Ups_P_1_2': 'first',
    'Firm_1_2_Ancillary': 'first',
    'Firm_1_2_P_I': 'first',
    'Carbon_Cost': 'first',
    'Site_Cost': 'first',
    'Scrap': 'first',
    "Renewable_Power": 'first',
    "Coal_Power": 'first',
    "Self_Power": 'first',
    'Natural_Gas_Power': 'first',
    "Nuclear_Power": 'first',
    'Hydro_Power': 'first',
    'employees': 'first',
    'Ups_P': 'first',
    'smelt_fuel': 'first',
    'total_site_1': 'first',
    'total_site_2': 'first',
})

df_p = df_p_agg.pivot_table(
    index=['Asset', 'Year', 'Product'],
    columns='Variable',
    values='Value',
    aggfunc='sum',
    fill_value=0
).reset_index()

df_p.columns.name = None

metadata_cols = df_p_agg[['Asset', 'Year', 'Product', 'Commodity', 'Owners', 'Asset - Operator',
                          'Operating status', 'Country', 'Asset - Region', 'Asset - State',
                          'Casting_1', 'Casting_F', 'Casting_2', 'Casting_P', 'Firm_Ups_1_2_I',
                          'Firm_Ups_1', 'Firm_Ups_2', 'Casting_Prod', 'Primary_Prod',
                          'Firm_Ups_1_2_3_I', 'total_site_1', 'total_site_2',
                          'Casting_1_2_I', 'Ups_P_1_2', 'Firm_1_2_Ancillary', 'Firm_1_2_P_I',
                          'Scrap', 'Carbon_Cost', 'Site_Cost', "Renewable_Power", "Coal_Power",
                          "Self_Power", 'Natural_Gas_Power', "Hydro_Power", 'Nuclear_Power',
                          'employees', 'Potroom_P', 'Ups_P',
                          'smelt_fuel']].drop_duplicates(subset=['Asset', 'Year', 'Product'])

df_p = df_p.merge(metadata_cols, on=['Asset', 'Year', 'Product'], how='left')

valid_products = ['Continuous cast rod', 'Primary ingot', 'Extrusion billet',
                  'Foundry alloy', 'Rolling slab', 'Continuous cast strip']

df_p_all = df_p.copy()

df_p = df_p[df_p['Product'].isin(valid_products)]
df_p = df_p[df_p['Production'] != 0]

df_p_calc = df_p.copy()

intensity_products = ['Continuous cast rod', 'Continuous cast strip',
                      'Extrusion billet', 'Rolling slab']
ingot_products = ['Primary ingot', 'Foundry alloy']

mask_intensity = df_p_calc['Product'].isin(intensity_products)
mask_ingot = df_p_calc['Product'].isin(ingot_products)

df_p_calc.loc[mask_intensity, 'Power'] = (
    df_p_calc.loc[mask_intensity, 'Power_P'] *
    df_p_calc.loc[mask_intensity, 'Production'] * 1000
)
df_p_calc.loc[mask_intensity, 'Fuel'] = (
    df_p_calc.loc[mask_intensity, 'Fuel_P'] *
    df_p_calc.loc[mask_intensity, 'Production'] * 1000
)

df_p_calc.loc[mask_ingot, 'Power'] = df_p_calc.loc[mask_ingot, 'Power Caster']
df_p_calc.loc[mask_ingot, 'Power_P'] = (
    df_p_calc.loc[mask_ingot, 'Power Caster'] /
    (df_p_calc.loc[mask_ingot, 'Production'] * 1000)
).replace([np.inf, -np.inf], 0).fillna(0)
df_p_calc.loc[mask_ingot, 'Fuel_P'] = 0

df_p_calc['Casting_F_safe'] = df_p_calc['Casting_F'].replace(0, np.nan)
df_p_calc['Casting_P_safe'] = df_p_calc['Casting_P'].replace(0, np.nan)
df_p_calc['Production_safe'] = df_p_calc['Production'].replace(0, np.nan)

df_p['Power'] = df_p_calc['Power']
df_p['Fuel'] = df_p_calc['Fuel']
df_p['Power_P'] = df_p_calc['Power_P']

df_p['Power_EF'] = df_p['Casting_2'] / df_p['Casting_P'].replace(0, np.nan)

df_p["Casting_1_I"] = (
    ((df_p_calc["Fuel"] / df_p_calc['Casting_F_safe']) *
     df_p_calc["Casting_1"]) /
    df_p_calc["Production_safe"]
).fillna(0).replace([np.inf, -np.inf], 0)

df_p["Casting_2_I"] = (
    (df_p_calc["Power"] / df_p_calc['Casting_P_safe']) *
    df_p_calc["Casting_2"] /
    df_p_calc["Production_safe"]
).fillna(0).replace([np.inf, -np.inf], 0)

df_p["Prod_Ups_1_2_I"] = (
    (df_p_calc["Firm_Ups_1_2_I"] * df_p_calc["Primary metal"]) /
    df_p_calc["Production_safe"]
).fillna(0).replace([np.inf, -np.inf], 0)

df_p["Prod_Ups_1_I"] = (
    ((df_p_calc["Firm_Ups_1"] / df_p_calc["Primary_Prod"]) * df_p_calc["Primary metal"]) /
    df_p_calc["Production_safe"]
).fillna(0).replace([np.inf, -np.inf], 0)

df_p["Prod_Ups_2_I"] = (
    ((df_p_calc["Firm_Ups_2"] / df_p_calc["Primary_Prod"]) * df_p_calc["Primary metal"]) /
    df_p_calc["Production_safe"]
).fillna(0).replace([np.inf, -np.inf], 0)

df_p["Prod_1_2_I"] = df_p["Casting_1_I"] + df_p["Casting_2_I"] + df_p["Prod_Ups_1_2_I"]
df_p["Prod_1_2"] = df_p["Prod_1_2_I"] * df_p["Production"]
df_p["Prod_1_I"] = df_p["Casting_1_I"] + df_p["Prod_Ups_1_I"]
df_p["Prod_2_I"] = df_p["Casting_2_I"] + df_p["Prod_Ups_2_I"]

df_p["Indst_1_2_I"] = (df_p_calc["Firm_Ups_1_2_I"] - df_p_calc["Ups_P_1_2"]).fillna(0)

df_p['Carbon_Costs?'] = np.where(
    df_p["Carbon_Cost"] > 0,
    'Face Carbon Costs',
    'Do Not Face Carbon Costs'
)

df_p["Clean_Power"] = df_p["Nuclear_Power"] + df_p["Hydro_Power"] + df_p["Renewable_Power"]
df_p["Carbon_Power"] = df_p["Coal_Power"] + df_p["Natural_Gas_Power"]
df_p["Total_Power"] = df_p["Clean_Power"] + df_p["Carbon_Power"]

df_p['Comp_Power'] = (df_p['Potroom_P'] + df_p_calc["Power"]) / df_p_calc['Production_safe']

df_p["Clean_Power_%"] = df_p["Clean_Power"] / df_p["Total_Power"]

df_p['Power_Up'] = df_p['Ups_P'] * df_p['Primary metal'] * 1000

smelt_fuel_int = (df_p['smelt_fuel'] /
                  (df_p['Primary_Prod'] * 1000).replace(0, np.nan))     # GJ/t primary
df_p['Fuel_Up'] = (smelt_fuel_int * df_p['Primary metal'] * 1000).fillna(0)

df_p['Power'] = df_p['Power'] + df_p['Power_Up']
df_p['Fuel'] = df_p['Fuel'] + df_p['Fuel_Up']

df_p['Power_I'] = df_p['Power'] / (df_p['Production'] * 1000)
df_p['Fuel_I'] = df_p['Fuel'] / (df_p['Production'] * 1000)

df_p['scope_1'] = df_p['Prod_1_I'] * df_p['Production']
df_p['scope_2'] = df_p['Prod_2_I'] * df_p['Production']

df_p.drop(columns=['Business costs', 'Capacity', 'Total CO2e emissions costs', 'Site costs',
                   'Primary metal', 'Power Caster', 'Casting_1', 'Casting_1', 'Casting_F',
                   'Casting_2', 'Casting_P', 'Firm_Ups_1_2_I', 'Firm_Ups_1', 'Firm_Ups_2',
                   'Casting_Prod', 'Primary_Prod', 'Firm_Ups_1_2_3_I', 'Casting_1_2_I', 'Power_P',
                   'Ups_P_1_2', 'Firm_1_2_Ancillary',  'Scrap', 'Carbon_Cost', 'Firm_1_2_P_I',
                   'Site_Cost', 'Potroom_P', 'Power_EF', 'Casting_1_I', 'Casting_2_I', 'Fuel_Up',
                   'Prod_Ups_1_2_I', 'Prod_Ups_1_I', 'Prod_Ups_2_I', 'Carbon_Costs?', 'Clean_Power',
                   'Carbon_Power', 'Total_Power', 'Comp_Power', 'Clean_Power_%', 'Fuel_P', 'Ups_P',
                   'Primary Production', 'Prod_1_2', 'Power_Up', 'smelt_fuel'], inplace=True)

df_p.rename(columns={'Asset - Operator': 'Operator',
                     'Asset - Region': 'Region',
                     'Asset - State': 'State',
                     'Production': 'production',
                     'Prod_1_2_I': 'scope_1_2_intensity',
                     'Prod_1_I': 'scope_1_intensity',
                     'Prod_2_I': 'scope_2_intensity',
                     'Power': 'Power_req',
                     'Fuel': 'Energy_req'
                     }, inplace=True)

df_p = df_p.sort_values(by=['Owners', 'Product'])
df_p.to_csv("unwrought_aluminum_clean.csv", index=False)
print("Data saved to unwrought_aluminum_clean.csv")
