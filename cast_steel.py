"""Cast steel - reconstructed final version.

Built on 15 Nov 2026 from cast_steel_stage1_v2.py. Changes are marked [RECONSTRUCTED]:
relative input path, operating/2006-2024 filter instead of 2024 only, and the final output
schema (power mix by source, State, renamed columns, absolute emissions). Known issues K12, K14,
K19 and K22 are left unfixed so the output can be checked against cast_steel_clean.csv (14 Oct).
"""
import pandas as pd
import re
import numpy as np

def cast_product(p):
    if p in ['Slab', 'Billet', 'Bloom', 'Ingot']:
        return p
    return None

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

def efficient_row_to_col_batch(df, variable_mappings):
    all_asset_years = df[['Asset', 'Year']].drop_duplicates()
    variables_to_pivot = list(variable_mappings.keys())
    pivot_data = df[df['Variable'].isin(variables_to_pivot)].copy()
    if pivot_data.empty:
        return df
    pivot_table = pivot_data.pivot_table(
        index=['Asset', 'Year'],
        columns='Variable',
        values='Value',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    column_rename_map = {}
    for old_var, new_col in variable_mappings.items():
        if old_var in pivot_table.columns:
            column_rename_map[old_var] = new_col
    pivot_table.rename(columns=column_rename_map, inplace=True)
    pivot_table = all_asset_years.merge(pivot_table, on=['Asset', 'Year'], how='left')
    new_columns = list(variable_mappings.values())
    for col in new_columns:
        if col in pivot_table.columns:
            pivot_table[col] = pivot_table[col].fillna(0)
    df_remaining = df[~df['Variable'].isin(variables_to_pivot)].copy()
    df_result = df_remaining.merge(pivot_table, on=['Asset', 'Year'], how='left')
    return df_result

# Load and initial processing
df = pd.read_csv("Commodities/steel.csv")
# [RECONSTRUCTED] Keep operating assets and historical years (2006-2024) before melting.
df = df[df['Operating status'] == 'Operating']
df = df.drop(columns=[c for c in df.columns if c.isdigit() and int(c) > 2024])
df.drop(columns=['Asset - Area', 'Asset - Basin name', 'Asset - Coalfield name',
                 'Asset - Technology', 'Asset - Mine type', 'Asset ID (select all)',
                 'Access codes (select all)', 'Scale', 'Asset - Service',
                 'Asset - Ore mineralisation', 'Category'],
        inplace=True)

df = df.melt(id_vars=['Commodity', 'Asset', 'Asset - Owners', 'Asset - Operator',
                      'Asset - Region', 'Asset - State',
                      'Variable', 'Product', 'Operating status', 'Country', 'Units'],
             var_name="Year", value_name="Value")

# [RECONSTRUCTED] The 2024-only filter is removed: the final output covers 2006-2024.

# Rename treated slab variants
treated_slab_mask = df['Product'] == 'Treated slab'
hrc_mask = df['Variable'].str.contains(r'for\s+HRC', case=False, regex=True)
plate_mask = df['Variable'].str.contains(r'for\s+plate', case=False, regex=True)
df.loc[treated_slab_mask & hrc_mask, 'Product'] = 'Treated Slab for HRC'
df.loc[treated_slab_mask & plate_mask, 'Product'] = 'Treated Slab for Plate'
df.loc[df['Variable'].str.contains("treated billet", case=False), "Product"] = "Treated billet"
df.loc[df['Variable'].str.contains("treated bloom", case=False), "Product"] = "Treated bloom"

# Variable aggregations
df = set_var_aggregation(df, ['Billet caster emissions, Scope 1, kt CO2e',
                              'Bloom caster emissions, Scope 1, kt CO2e',
                              'Slab caster emissions, Scope 1, kt CO2e',
                              'Ingot caster emissions, Scope 1, kt CO2e'],
                         ['Asset', 'Product', 'Year'], 'Caster Scope 1, kt CO2e')

df = set_var_aggregation(df, ['Treated slab for plate emissions, Scope 1, kt CO2e',
                              'Treated slab for HRC emissions, Scope 1, kt CO2e',
                              'Treated billet emissions, Scope 1, kt CO2e',
                              'Treated bloom emissions, Scope 1, kt CO2e'],
                         ['Asset', 'Product', 'Year'], 'Treatment Scope 1, kt CO2e')

df = set_var_aggregation(df, ['Production, billet, Mt billet',
                              'Production, bloom, Mt bloom',
                              'Production, slab, Mt slab',
                              'Production, ingot, Mt ingot'],
                         ['Asset', 'Product', 'Year'], 'Production, rolled product, Mt')

df = set_var_aggregation(df, ['Treated slab for plate, Mt',
                              'Treated slab for HRC, Mt',
                              'Treated billet, Mt',
                              'Treated bloom, Mt'],
                         ['Asset', 'Product', 'Year'], 'Production, treated product, Mt')

df = set_var_aggregation(df, ['Power - ancillary input, kWh/t billet',
                              'Power input, kWh/t billet',
                              'Power - ancillary input, kWh/t bloom',
                              'Power input, kWh/t bloom',
                              'Power - ancillary input, kWh/t slab',
                              'Power input, kWh/t slab',
                              'Power - ancillary input, kWh/t ingot',
                              'Power input, kWh/t ingot'],
                         ['Asset', 'Product', 'Year'], 'Power, kWh/t rolled product')

df = set_var_aggregation(df, ["Total process energy requirement, GJ/t billet",
                              "Total process energy requirement, GJ/t bloom",
                              "Total process energy requirement, GJ/t slab"],
                         ['Asset', 'Product', 'Year'], 'Energy, GJ/t rolled product')

df = set_var_aggregation(df, ['Power - ancillary input, kWh/t treated slab (for plate)',
                              'Power input, kWh/t treated slab (for plate)',
                              'Power - ancillary input, kWh/t treated slab (for HRC)',
                              'Power input, kWh/t treated slab (for HRC)',
                              'Power - ancillary input, kWh/t treated billet',
                              'Power input, kWh/t treated billet',
                              'Power - ancillary input, kWh/t treated bloom',
                              'Power input, kWh/t treated bloom'],
                         ['Asset', 'Product', 'Year'], 'Treatment Power, kWh/t treated product')

df = set_var_aggregation(df, ['Total costs, $/t billet',
                              'Total costs, $/t bloom',
                              'Total costs, $/t slab',
                              'Total costs, $/t ingot'],
                         ['Asset', 'Product', 'Year'], 'Total Costs, $/t rolled product')

df = set_var_aggregation(df, ['CO2 costs, $/t billet',
                              'CO2 costs, $/t bloom',
                              'CO2 costs, $/t slab',
                              'CO2 costs, $/t ingot'],
                         ['Asset', 'Product', 'Year'], 'CO2 Costs, $/t rolled product')

df = set_var_aggregation(df, ['Total costs, $/t treated slab (for plate)',
                              'Total costs, $/t treated slab (for HRC)',
                              'Total costs, $/t treated billet',
                              'Total costs, $/t treated bloom'],
                         ['Asset', 'Product', 'Year'], 'Total Treatment Costs, $/t treated product')

df = set_var_aggregation(df, ['Sales, billet, Mt',
                              'Sales, bloom, Mt',
                              'Sales, slab, Mt'],
                         ['Asset', 'Product', 'Year'], 'Sales, rolled product, Mt')

df = set_var_aggregation(df, ['Capacity, billet, Mt billet',
                              'Capacity, bloom, Mt bloom',
                              'Capacity, slab, Mt slab',
                              'Capacity, ingot, Mt ingot'],
                         ['Asset', 'Product', 'Year'], 'Capacity, rolled product, Mt')

df = set_var_aggregation(df, ['EAF liquid steel input, kg/t billet',
                              'EAF liquid steel input, kg/t bloom',
                              'EAF liquid steel input, kg/t slab'],
                         ['Asset', 'Product', 'Year'], 'EAF ls input, kg/t rolled product')

df = set_var_aggregation(df, ['BOF liquid steel input, kg/t billet',
                              'BOF liquid steel input, kg/t bloom',
                              'BOF liquid steel input, kg/t ingot',
                              'BOF liquid steel input, kg/t slab'],
                         ['Asset', 'Product', 'Year'], 'BOF ls input, kg/t rolled product')

df = set_var_aggregation(df, ['Custom steelmaking liquid steel input, kg/t billet',
                              'Custom steelmaking liquid steel input, kg/t slab'],
                         ['Asset', 'Product', 'Year'],
                         'Custom steelmaking ls input, kg/t rolled product')

df = set_var_aggregation(df, ['Conarc liquid steel input, kg/t billet',
                              'Conarc liquid steel input, kg/t slab'],
                         ['Asset', 'Product', 'Year'], 'Conarc ls input, kg/t rolled product')

df['Owners'] = df['Asset - Owners'].apply(get_majority_owner).fillna('Unknown')



df = df[df['Variable'].isin(['Electric arc furnace emissions, Scope 1, kt CO2e',
                             'Electric arc furnace emissions, Scope 2, kt CO2e',
                             'Longs emissions, Scope 1, kt CO2e',
                             'Longs emissions, Scope 2, kt CO2e',
                             'Flats emissions, Scope 1, kt CO2e',
                             'Flats emissions, Scope 2, kt CO2e',
                             'Basic oxygen furnace emissions, Scope 1, kt CO2e',
                             'Basic oxygen furnace emissions, Scope 2, kt CO2e',
                             'Custom steelmaking emissions, Scope 1, kt CO2e',
                             'Custom steelmaking emissions, Scope 2, kt CO2e',
                             'Conarc emissions, Scope 1, kt CO2e',
                             'Conarc emissions, Scope 2, kt CO2e',
                             'Casters emissions, Scope 2, kt CO2e',
                             'Production, EAF steel, Mt liquid steel',
                             'Production, BOF steel, Mt liquid steel',
                             'Production, custom steel, Mt liquid steel',
                             'Production, Conarc steel, Mt liquid steel',
                             'Slab input, t/t treated slab (for plate)',
                             'Slab input, t/t treated slab (for HRC)',
                             'Billet input, t/t treated billet',
                             'Bloom input, t/t treated bloom',
                             'Caster Scope 1, kt CO2e',
                             'Treatment Scope 1, kt CO2e',
                             'Production, rolled product, Mt',
                             'Production, treated product, Mt',
                             'Power, kWh/t rolled product',
                             'Energy, GJ/t rolled product',
                             'Treatment Power, kWh/t treated product',
                             'Total Costs, $/t rolled product',
                             'Total Treatment Costs, $/t treated product',
                             'Capacity, rolled product, Mt',
                             'EAF ls input, kg/t rolled product',
                             'BOF ls input, kg/t rolled product',
                             'Custom steelmaking ls input, kg/t rolled product',
                             'Conarc ls input, kg/t rolled product',
                             'Power required, site, GWh',
                             'Self-generated power, MWh',
                             'SSPA scrap charge, %',
                             'CO2 Costs, $/t rolled product',
                             'Electric arc furnace cumulative product emissions, Scope 1, t CO2e/t liquid steel',
                             'Basic oxygen furnace cumulative product emissions, Scope 1, t CO2e/t liquid steel',
                             'Custom steelmaking cumulative product emissions, Scope 1, t CO2e/t liquid steel',
                             'Conarc cumulative product emissions, Scope 1, t CO2e/t liquid steel',
                             'Electric arc furnace cumulative product emissions, Scope 1 & 2, t CO2e/t liquid steel',
                             'Basic oxygen furnace cumulative product emissions, Scope 1 & 2, t CO2e/t liquid steel',
                             'Custom steelmaking cumulative product emissions, Scope 1 & 2, t CO2e/t liquid steel',
                             'Conarc cumulative product emissions, Scope 1 & 2, t CO2e/t liquid steel',
                             'Employees, total incl. contractors, #'])]

df['Variable'] = df['Variable'].str.rsplit(',', n=1).str[0].str.strip()

#external_scrap_mask = df['Variable'] == 'External scrap input'
#df.loc[external_scrap_mask, 'Variable'] = 'External scrap input - ' + df.loc[external_scrap_mask, 'Product']


# Unit conversions
mask = df['Variable'].str.contains('ls input', case=False, na=False)
df.loc[mask, 'Value'] = df.loc[mask, 'Value'] / 1000
mask = df['Variable'].str.contains('Production', case=False, na=False)
df.loc[mask, 'Value'] = df.loc[mask, 'Value'] * 1000
mask = df['Variable'].str.contains('Power required', case=False, na=False)
df.loc[mask, 'Value'] = df.loc[mask, 'Value'] * 1000
#mask = df['Variable'].str.contains('External scrap input', case=False, na=False)
#df.loc[mask, 'Value'] = df.loc[mask, 'Value'] / 1000

# Deduplication - Remove asset-level duplicates
asset_level_variables = [
    'Electric arc furnace emissions, Scope 1',
    'Electric arc furnace emissions, Scope 2',
    'Basic oxygen furnace emissions, Scope 1',
    'Basic oxygen furnace emissions, Scope 2',
    'Custom steelmaking emissions, Scope 1',
    'Custom steelmaking emissions, Scope 2',
    'Longs emissions, Scope 1',
    'Longs emissions, Scope 2',
    'Flats emissions, Scope 1',
    'Flats emissions, Scope 2',
    'Conarc emissions, Scope 1',
    'Conarc emissions, Scope 2',
    'Electric arc furnace cumulative product emissions, Scope 1 & 2',
    'Basic oxygen furnace cumulative product emissions, Scope 1 & 2',
    'Custom steelmaking cumulative product emissions, Scope 1 & 2',
    'Conarc cumulative product emissions, Scope 1 & 2',
    'Electric arc furnace cumulative product emissions, Scope 1',
    'Basic oxygen furnace cumulative product emissions, Scope 1',
    'Custom steelmaking cumulative product emissions, Scope 1',
    'Conarc cumulative product emissions, Scope 1',
    'Production, EAF steel',
    'Production, BOF steel',
    'Production, custom steel',
    'Production, Conarc steel',
    'Power required, site',
    'Self-generated power',
    'SSPA scrap charge',
    'Employees, total incl. contractors'
]

product_level_variables = [
    'Casters emissions, Scope 2',
    'Slab input',
    'Billet input',
    'Bloom input',
    'Caster Scope 1',
    'Treatment Scope 1',
    'Production, rolled product',
    'Production, treated product',
    'CO2 Costs',
    'Power',
    'Energy',
    'Treatment Power',
    'Total Costs',
    'Total Treatment Costs',
    'Capacity, rolled product',
    'EAF ls input',
    'BOF ls input',
    'Custom steelmaking ls input',
    'Conarc ls input'
]

df_asset_level = df[df['Variable'].isin(asset_level_variables)].copy()
df_product_level = df[df['Variable'].isin(product_level_variables)].copy()

df_asset_level_dedup = df_asset_level.groupby(['Asset', 'Year', 'Variable'], as_index=False).first()
df = pd.concat([df_asset_level_dedup, df_product_level], ignore_index=True)

# Variable mappings
variable_mappings = {
    'Electric arc furnace emissions, Scope 1': 'Scope_1_EAF',
    'Electric arc furnace emissions, Scope 2': 'Scope_2_EAF',
    'Longs emissions, Scope 1': 'Scope_1_Longs',
    'Longs emissions, Scope 2': 'Scope_2_Longs',
    'Flats emissions, Scope 1': 'Scope_1_Flats',
    'Flats emissions, Scope 2': 'Scope_2_Flats',
    'Basic oxygen furnace emissions, Scope 1': 'Scope_1_BOF',
    'Basic oxygen furnace emissions, Scope 2': 'Scope_2_BOF',
    'Custom steelmaking emissions, Scope 1': 'Scope_1_CS',
    'Custom steelmaking emissions, Scope 2': 'Scope_2_CS',
    'Conarc emissions, Scope 1': 'Scope_1_Conarc',
    'Conarc emissions, Scope 2': 'Scope_2_Conarc',
    'Electric arc furnace cumulative product emissions, Scope 1 & 2': 'Scope_1_2_EAF_Cum',
    'Basic oxygen furnace cumulative product emissions, Scope 1 & 2': 'Scope_1_2_BOF_Cum',
    'Custom steelmaking cumulative product emissions, Scope 1 & 2': 'Scope_1_2_Custom_Cum',
    'Conarc cumulative product emissions, Scope 1 & 2': 'Scope_1_2_Conarc_Cum',
    'Electric arc furnace cumulative product emissions, Scope 1': 'Scope_1_EAF_Cum',
    'Basic oxygen furnace cumulative product emissions, Scope 1': 'Scope_1_BOF_Cum',
    'Custom steelmaking cumulative product emissions, Scope 1': 'Scope_1_Custom_Cum',
    'Conarc cumulative product emissions, Scope 1': 'Scope_1_Conarc_Cum',
    'Casters emissions, Scope 2': 'Scope_2_Casters',
    'Production, EAF steel': 'Production_EAF',
    'Production, BOF steel': 'Production_BOF',
    'Production, custom steel': 'Production_CS',
    'Production, Conarc steel': 'Production_Conarc',
    'Power required, site': 'Power_Required',
    'Self-generated power': 'Power_Self_Gen',
    'SSPA scrap charge': 'SSPA_Scrap',
    'Employees, total incl. contractors': 'employees'
}

print(df.columns)

df = efficient_row_to_col_batch(df, variable_mappings)
df = df[df['Operating status'] == 'Operating']

# Separate base and treated products
variable_mapping = {
    'Slab input': 'Slab_Input',
    'Billet input': 'Billet_Input',
    'Bloom input': 'Bloom_Input',
    'Caster Scope 1': 'Scope_1_Casters',
    'Treatment Scope 1': 'Scope_1_Treatment',
    'Production, rolled product': 'Production_caster',
    'Production, treated product': 'Production_treatment',
    'Power': 'Power',
    'Energy': 'Energy',
    'Treatment Power': 'Treat_Power',
    'Total Costs': 'Total_Costs',
    'Total Treatment Costs': 'Total_Treat_Costs',
    'Capacity, rolled product': 'Capacity_caster',
    'EAF ls input': 'EAF_LS_Input',
    'BOF ls input': 'BOF_LS_Input',
    'Custom steelmaking ls input': 'Custom_Steel_LS_Input',
    'Conarc ls input': 'Conarc_LS_Input',
    'CO2 Costs' : 'CO2_Costs'
}

df_treated_raw = df[df["Product"].isin([
    "Treated Slab for HRC", "Treated Slab for Plate",
    "Treated billet", "Treated bloom"
])].copy()

df_base_raw = df[df["Product"].isin(["Ingot", "Slab", "Billet", "Bloom"])].copy()

# Process base products
asset_columns = [
    'Commodity', 'Asset', 'Owners', 'Asset - Operator', 'Product',
    'Operating status', 'Country', 'Asset - Region', 'Year',
    'Scope_1_EAF', 'Scope_2_EAF', 'Scope_1_Longs', 'Scope_2_Longs',
    'Scope_1_Flats', 'Scope_2_Flats', 'Scope_1_BOF', 'Scope_2_BOF',
    'Scope_1_CS', 'Scope_2_CS', 'Scope_1_Conarc', 'Scope_2_Conarc',
    'Scope_2_Casters', 'Production_EAF', 'Production_BOF',
    'Production_CS', 'Production_Conarc', 'Power_Required',
    'Power_Self_Gen', 'Scope_1_2_EAF_Cum', 'Scope_1_2_BOF_Cum',
    'Scope_1_2_Custom_Cum', 'Scope_1_2_Conarc_Cum', 'Cast_Product',
    'Scope_1_EAF_Cum', 'Scope_1_BOF_Cum', 'Scope_1_Custom_Cum', 'Scope_1_Conarc_Cum',
    'SSPA_Scrap', 'employees'
]

if len(df_base_raw) > 0:
    df_p = df_base_raw.copy()
    df_p["Cast_Product"] = df_p["Product"].apply(cast_product)
    df_p["Variable"] = df_p["Variable"].map(variable_mapping).str.rsplit(",", n=1).str[0].str.strip()

    df_p['Owners'] = df_p['Owners'].fillna('Unknown')
    df_p['Asset - Operator'] = df_p['Asset - Operator'].fillna('Unknown')

    for col in asset_columns:
        if col in df_p.columns:
            if df_p[col].dtype == 'object':
                df_p[col] = df_p[col].fillna("Unknown")
            else:
                df_p[col] = df_p[col].fillna(0)

    df_p["Units"] = df_p["Units"].fillna("Unknown")
    df_p["Value"] = df_p["Value"].fillna(0)

    df_p = df_p.pivot_table(
        index=asset_columns,
        columns="Variable",
        values=["Value", "Units"],
        aggfunc={"Value": "sum", "Units": "first"},
        fill_value=0
    )

    df_p.columns = [vr if vl == "Value" else f"{vl}_{vr}" for vl, vr in df_p.columns]
    df_p = df_p.reset_index()
else:
    df_p = pd.DataFrame()

# Process treated products
if len(df_treated_raw) > 0:
    df_treated_raw["Variable"] = df_treated_raw["Variable"].map(variable_mapping).fillna(df_treated_raw["Variable"]).str.rsplit(",", n=1).str[0].str.strip()

    df_treated_raw['Owners'] = df_treated_raw['Owners'].fillna('Unknown')
    df_treated_raw['Asset - Operator'] = df_treated_raw['Asset - Operator'].fillna('Unknown')
    df_treated_raw["Units"] = df_treated_raw["Units"].fillna("Unknown")
    df_treated_raw["Value"] = df_treated_raw["Value"].fillna(0)

    df_treated_raw = df_treated_raw.rename(columns={"Product": "Treated_Product"})

    treated_asset_cols = [
        'Commodity', 'Asset', 'Owners', 'Asset - Operator',
        'Operating status', 'Country', 'Asset - Region', 'Year',
        'Scope_1_EAF', 'Scope_2_EAF', 'Scope_1_Longs', 'Scope_2_Longs',
        'Scope_1_Flats', 'Scope_2_Flats', 'Scope_1_BOF', 'Scope_2_BOF',
        'Scope_1_CS', 'Scope_2_CS', 'Scope_1_Conarc', 'Scope_2_Conarc',
        'Scope_2_Casters', 'Production_EAF', 'Production_BOF',
        'Production_CS', 'Production_Conarc', 'Power_Required',
        'Power_Self_Gen', 'Scope_1_2_EAF_Cum', 'Scope_1_2_BOF_Cum',
        'Scope_1_2_Custom_Cum', 'Scope_1_2_Conarc_Cum', 'Scope_1_EAF_Cum',
        'Scope_1_BOF_Cum', 'Scope_1_Custom_Cum', 'Scope_1_Conarc_Cum',
        'Treated_Product', 'SSPA_Scrap', 'employees'
    ]

    for col in treated_asset_cols:
        if col in df_treated_raw.columns:
            if df_treated_raw[col].dtype == 'object':
                df_treated_raw[col] = df_treated_raw[col].fillna("Unknown")
            else:
                df_treated_raw[col] = df_treated_raw[col].fillna(0)

    df_treated_agg = df_treated_raw.groupby(['Asset', 'Year', 'Treated_Product', 'Variable'], as_index=False).agg({
        'Value': 'sum', 'Units': 'first', 'Owners': 'first', 'Asset - Operator': 'first',
        'Commodity': 'first', 'Operating status': 'first', 'Country': 'first',
        'Asset - Region': 'first', 'Scope_1_EAF': 'first', 'Scope_2_EAF': 'first',
        'Scope_1_Longs': 'first', 'Scope_2_Longs': 'first', 'Scope_1_Flats': 'first',
        'Scope_2_Flats': 'first', 'Scope_1_BOF': 'first', 'Scope_2_BOF': 'first',
        'Scope_1_CS': 'first', 'Scope_2_CS': 'first', 'Scope_1_Conarc': 'first',
        'Scope_2_Conarc': 'first', 'Scope_2_Casters': 'first', 'Production_EAF': 'first',
        'Production_BOF': 'first', 'Production_CS': 'first', 'Production_Conarc': 'first',
        'Power_Required': 'first', 'Power_Self_Gen': 'first', 'Scope_1_2_EAF_Cum': 'first',
        'Scope_1_2_BOF_Cum': 'first', 'Scope_1_2_Custom_Cum': 'first',
        'Scope_1_2_Conarc_Cum': 'first', 'SSPA_Scrap': 'first', 'Scope_1_EAF_Cum': 'first',
        'Scope_1_BOF_Cum': 'first', 'Scope_1_Custom_Cum': 'first',
        'Scope_1_Conarc_Cum': 'first', 'employees': 'first'
    })

    treated_index_cols = ['Asset', 'Year', 'Treated_Product', 'Owners', 'Asset - Operator',
                          'Commodity', 'Operating status', 'Country', 'Asset - Region',
                          'Scope_1_EAF', 'Scope_2_EAF', 'Scope_1_Longs', 'Scope_2_Longs',
                          'Scope_1_Flats', 'Scope_2_Flats', 'Scope_1_BOF', 'Scope_2_BOF',
                          'Scope_1_CS', 'Scope_2_CS', 'Scope_1_Conarc', 'Scope_2_Conarc',
                          'Scope_2_Casters', 'Production_EAF', 'Production_BOF',
                          'Production_CS', 'Production_Conarc', 'Power_Required',
                          'Power_Self_Gen', 'Scope_1_2_EAF_Cum', 'Scope_1_2_BOF_Cum',
                          'Scope_1_2_Custom_Cum', 'Scope_1_2_Conarc_Cum', 'Scope_1_EAF_Cum',
                          'Scope_1_BOF_Cum','Scope_1_Custom_Cum', 'Scope_1_Conarc_Cum',
                          'SSPA_Scrap', 'employees']

    df_treated = df_treated_agg.pivot_table(
        index=treated_index_cols,
        columns='Variable',
        values='Value',
        aggfunc='first',
        fill_value=0
    ).reset_index()

    df_treated.columns.name = None
    df_treated = df_treated[df_treated.get('Production_treatment', 0) > 0].copy()
else:
    df_treated = pd.DataFrame()

if 'Asset - Owners' in df_p.columns:
    df_p = df_p.drop(columns=['Asset - Owners'])
if 'Asset - Owners' in df_treated.columns:
    df_treated = df_treated.drop(columns=['Asset - Owners'])

# Merge base and treated products
product_treated_map = {
    "Slab": ["Treated Slab for HRC", "Treated Slab for Plate"],
    "Billet": ["Treated billet"],
    "Bloom": ["Treated bloom"],
    "Ingot": []
}

BASE_SCHEMA_DEFAULTS = {
    'Cast_Product': None,
    'Production_caster': 0,
    'Scope_1_Casters': 0,
    'Power': 0,
    'Energy': 0,
    'Total_Costs': 0,
    'Capacity_caster': 0,
    'EAF_LS_Input': 0,
    'BOF_LS_Input': 0,
    'Custom_Steel_LS_Input': 0,
    'Conarc_LS_Input': 0,
    'CO2_Costs': 0,
}

treated_cols = [c for c in df_treated.columns if c not in [
    'Asset', 'Year', 'Treated_Product', 'Owners', 'Asset - Operator',
    'Commodity', 'Operating status', 'Country', 'Asset - Region',
    'Scope_1_EAF', 'Scope_2_EAF', 'Scope_1_Longs', 'Scope_2_Longs',
    'Scope_1_Flats', 'Scope_2_Flats', 'Scope_1_BOF', 'Scope_2_BOF',
    'Scope_1_CS', 'Scope_2_CS', 'Scope_1_Conarc', 'Scope_2_Conarc',
    'Scope_2_Casters', 'Production_EAF', 'Production_BOF',
    'Production_CS', 'Production_Conarc', 'Power_Required',
    'Scope_1_2_EAF_Cum', 'Scope_1_2_BOF_Cum', 'Scope_1_2_Custom_Cum',
    'Scope_1_2_Conarc_Cum', 'Power_Self_Gen', 'Scope_1_EAF_Cum',
    'Scope_1_BOF_Cum', 'Scope_1_Custom_Cum', 'Scope_1_Conarc_Cum',
    'SSPA_Scrap', 'employees'
]]

base_assets = set(df_p['Asset'].unique()) if len(df_p) > 0 else set()
treated_assets = set(df_treated['Asset'].unique())

matched_treated_products = set()
all_rows = []

# Merge base with treated
if len(df_p) > 0:
    for _, base_row in df_p.iterrows():
        base_product = base_row['Product']
        asset = base_row['Asset']
        year = base_row['Year']
        expected_treated = product_treated_map.get(base_product, [])

        if not expected_treated:
            all_rows.append(base_row)
            continue

        matching_treated = df_treated[
            (df_treated['Asset'] == asset) &
            (df_treated['Year'] == year) &
            (df_treated['Treated_Product'].isin(expected_treated))
        ]

        if matching_treated.empty:
            all_rows.append(base_row)
        else:
            for _, treated_row in matching_treated.iterrows():
                merged = base_row.copy()
                merged['Treated_Product'] = treated_row['Treated_Product']

                for col in treated_cols:
                    if col in treated_row.index:
                        merged[col] = treated_row.get(col, 0)

                all_rows.append(merged)
                matched_treated_products.add((asset, year, treated_row['Treated_Product']))

# Add treated-only assets
for _, treated_row in df_treated.iterrows():
    asset = treated_row['Asset']
    year = treated_row['Year']
    treated_prod = treated_row['Treated_Product']

    key = (asset, year, treated_prod)

    if key in matched_treated_products:
        continue

    if asset not in base_assets:
        new_row = {}
        metadata_cols = ['Asset', 'Year', 'Owners', 'Asset - Operator',
                         'Commodity', 'Operating status', 'Country', 'Asset - Region',
                         'Scope_1_EAF', 'Scope_2_EAF', 'Scope_1_Longs', 'Scope_2_Longs',
                         'Scope_1_Flats', 'Scope_2_Flats', 'Scope_1_BOF', 'Scope_2_BOF',
                         'Scope_1_CS', 'Scope_2_CS', 'Scope_1_Conarc', 'Scope_2_Conarc',
                         'Scope_2_Casters', 'Production_EAF', 'Production_BOF',
                         'Production_CS', 'Production_Conarc', 'Power_Required',
                         'Power_Self_Gen', 'Scope_1_2_EAF_Cum', 'Scope_1_2_BOF_Cum',
                         'Scope_1_2_Custom_Cum', 'Scope_1_2_Conarc_Cum', 'Scope_1_EAF_Cum',
                         'Scope_1_BOF_Cum', 'Scope_1_Custom_Cum', 'Scope_1_Conarc_Cum',
                         'SSPA_Scrap', 'employees']

        for col in metadata_cols:
            default_val = 0 if col not in ['Asset', 'Year', 'Owners', 'Asset - Operator',
                                            'Commodity', 'Operating status', 'Country', 'Asset - Region'] else 'Unknown'
            new_row[col] = treated_row.get(col, default_val)

        new_row['Product'] = treated_prod
        new_row['Treated_Product'] = treated_prod
        new_row['Cast_Product'] = None

        for col, default in BASE_SCHEMA_DEFAULTS.items():
            new_row[col] = default

        for col in treated_cols:
            if col in treated_row.index:
                new_row[col] = treated_row[col]

        all_rows.append(new_row)

if len(all_rows) > 0:
    df_p = pd.DataFrame(all_rows)
else:
    df_p = pd.DataFrame()

if 'Treated_Product' not in df_p.columns:
    df_p['Treated_Product'] = None

df_p = df_p[(df_p.get('Production_caster', 0) > 0) | (df_p.get('Production_treatment', 0) > 0)]

# Final aggregation
groupby_cols = [c for c in asset_columns if c in df_p.columns]
if 'Treated_Product' in df_p.columns:
    groupby_cols.append('Treated_Product')

df_p = df_p.groupby(groupby_cols, dropna=False).first().reset_index()
df_p = df_p.sort_values(by=["Owners", "Product"])
df_p = df_p[(df_p.get('Production_caster', 0) != 0) | (df_p.get('Production_treatment', 0) != 0)]

# ============================================================================
# NEW: ADD UPSTREAM STEP COUNTS (OPTIMIZED WITH VECTORIZATION)
# ============================================================================

print("Starting upstream step counting...")

# Re-read unfiltered data once
df_unfiltered = pd.read_csv("Commodities/steel.csv")

# Melt to get asset-year-variable structure
df_unfiltered_melted = df_unfiltered.melt(
    id_vars=['Commodity', 'Asset', 'Asset - Owners', 'Asset - Operator',
             'Asset - Region', 'Asset - State', 'Variable', 'Product',
             'Operating status', 'Country', 'Units'],
    var_name="Year", value_name="Value"
)

# Convert Value to numeric
df_unfiltered_melted['Value'] = pd.to_numeric(df_unfiltered_melted['Value'], errors='coerce')
df_unfiltered_melted['Value'] = df_unfiltered_melted['Value'].fillna(0)

# Define patterns for each upstream step type
raw_materials_pattern = r'(sinter|pellet|coke|burnt|dolomite|lime|gas)'
intermediate_pattern = r'(hot metal|dri(?!\/)|\bblast furnace|hbi)'
steelmaking_pattern = r'(eaf steel|bof steel|custom steel|conarc)'

# Create boolean masks using str.contains (vectorized)
df_unfiltered_melted['Variable_lower'] = df_unfiltered_melted['Variable'].str.lower()
df_unfiltered_melted['is_raw_materials'] = df_unfiltered_melted['Variable_lower'].str.contains(raw_materials_pattern, na=False, regex=True)
df_unfiltered_melted['is_intermediate'] = df_unfiltered_melted['Variable_lower'].str.contains(intermediate_pattern, na=False, regex=True)
df_unfiltered_melted['is_steelmaking'] = df_unfiltered_melted['Variable_lower'].str.contains(steelmaking_pattern, na=False, regex=True)

# Filter for non-zero values with each step type
df_raw_nonzero = df_unfiltered_melted[(df_unfiltered_melted['is_raw_materials']) & (df_unfiltered_melted['Value'] > 0)]
df_inter_nonzero = df_unfiltered_melted[(df_unfiltered_melted['is_intermediate']) & (df_unfiltered_melted['Value'] > 0)]

# For steelmaking, also check the filtered df (smaller dataset)
df['Variable_lower'] = df['Variable'].str.lower()
df['is_steelmaking'] = df['Variable_lower'].str.contains(steelmaking_pattern, na=False, regex=True)
df_steel_nonzero = df[(df['is_steelmaking']) & (df['Value'] > 0)]

# Convert to sets for O(1) lookup
raw_asset_years = set(df_raw_nonzero[['Asset', 'Year']].drop_duplicates().apply(tuple, axis=1))
inter_asset_years = set(df_inter_nonzero[['Asset', 'Year']].drop_duplicates().apply(tuple, axis=1))
steel_asset_years = set(df_steel_nonzero[['Asset', 'Year']].drop_duplicates().apply(tuple, axis=1))

print(f"  Raw materials found in {len(raw_asset_years)} asset-year combinations")
print(f"  Intermediate found in {len(inter_asset_years)} asset-year combinations")
print(f"  Steelmaking found in {len(steel_asset_years)} asset-year combinations")

# Create lookup dataframe with upstream step counts
upstream_data = []
for asset in df_p['Asset'].unique():
    for year in df_p['Year'].unique():
        key = (asset, str(int(year)))
        steps = 0
        if key in raw_asset_years:
            steps += 1
        if key in inter_asset_years:
            steps += 1
        if key in steel_asset_years:
            steps += 1
        upstream_data.append({'Asset': asset, 'Year': str(int(year)), 'upstream_steps': steps})

df_upstream_map = pd.DataFrame(upstream_data)

# Merge with df_p
df_p['Year'] = df_p['Year'].astype(str)
df_p = df_p.merge(df_upstream_map, on=['Asset', 'Year'], how='left')
df_p['upstream_steps'] = df_p['upstream_steps'].fillna(0).astype(int)

# Calculate final step counts (vectorized)
df_p['Upstream_Steps_Caster'] = (
    df_p['upstream_steps'] +
    (df_p['Production_caster'] > 0).astype(int)
)

df_p['Upstream_Steps_Treatment'] = (
    df_p['upstream_steps'] +
    (df_p['Production_caster'] > 0).astype(int) +
    (df_p['Production_treatment'] > 0).astype(int)
)

# Drop intermediate columns
df_p = df_p.drop(columns=['upstream_steps'])

print("Upstream step counting complete!")

# ============================================================================
# Carbon intensity calculations
df_p["SM_CS_I_1_2"] = np.where(df_p["Production_CS"] != 0, (df_p["Scope_1_CS"] + df_p["Scope_2_CS"]) / df_p["Production_CS"], 0)
df_p["SM_EAF_I_1_2"] = np.where(df_p["Production_EAF"] != 0, (df_p["Scope_1_EAF"] + df_p["Scope_2_EAF"]) / df_p["Production_EAF"], 0)
df_p["SM_BOF_I_1_2"] = np.where(df_p["Production_BOF"] != 0, (df_p["Scope_1_BOF"] + df_p["Scope_2_BOF"]) / df_p["Production_BOF"], 0)
df_p["SM_CA_I_1_2"] = np.where(df_p["Production_Conarc"] != 0, (df_p["Scope_1_Conarc"] + df_p["Scope_2_Conarc"]) / df_p["Production_Conarc"], 0)

df_p["O_P_Prc"] = (df_p["Power_Required"] - df_p["Power_Self_Gen"]) / df_p["Power_Required"]

df_p["Cast_Power_MW"] = df_p["Power"] * df_p["Production_caster"]
df_p["Treat_Power_MW"] = df_p["Treat_Power"] * df_p["Production_treatment"]

cast_power_by_asset = df_p.groupby("Asset")["Cast_Power_MW"].sum().reset_index()
treat_power_by_asset = df_p.groupby("Asset")["Treat_Power_MW"].sum().reset_index()
power_by_asset = cast_power_by_asset.merge(treat_power_by_asset, on="Asset", how="outer", suffixes=("_cast", "_treat")).fillna(0)
power_by_asset["Power_total"] = power_by_asset["Cast_Power_MW"] + power_by_asset["Treat_Power_MW"]
df_p = df_p.merge(power_by_asset[["Asset", "Power_total"]], on="Asset", how="left")

df_p["O_Cast_power_prc"] = (df_p["Cast_Power_MW"] / df_p["Power_total"]) * df_p["O_P_Prc"]
df_p["O_Treat_power_prc"] = (df_p["Treat_Power_MW"] / df_p["Power_total"]) * df_p["O_P_Prc"]

df_p["Cast_2"] = df_p["Scope_2_Casters"] * df_p["O_Cast_power_prc"]
df_p["Cast_1_2"] = df_p["Scope_1_Casters"] + df_p["Cast_2"]
df_p["Cast_1_I"] = np.where(df_p["Production_caster"] != 0, df_p["Scope_1_Casters"] / df_p["Production_caster"], 0)
df_p["Cast_2_I"] = np.where(df_p["Production_caster"] != 0, df_p["Cast_2"] / df_p["Production_caster"], 0)
df_p["Cast_I_1_2"] = np.where(df_p["Production_caster"] != 0, df_p["Cast_1_2"] / df_p["Production_caster"], 0)

df_p["Treat_2"] = + df_p["Scope_2_Casters"] * df_p["O_Treat_power_prc"]
df_p["Treat_1_2"] = df_p["Scope_1_Treatment"] + df_p["Treat_2"]
df_p["Treat_1_I"] = np.where(df_p["Production_treatment"] != 0, df_p["Scope_1_Treatment"] / df_p["Production_treatment"], 0)
df_p["Treat_2_I"] = np.where(df_p["Production_treatment"] != 0, df_p["Treat_2"] / df_p["Production_treatment"], 0)
df_p["Treat_I_1_2"] = np.where(df_p["Production_treatment"] != 0, df_p["Treat_1_2"] / df_p["Production_treatment"], 0)

df_p["Cast_Inputs_I_1_2"] = (
    df_p["Scope_1_2_EAF_Cum"] * df_p["EAF_LS_Input"]
    + df_p["Scope_1_2_Conarc_Cum"] * df_p["Conarc_LS_Input"]
    + df_p["Scope_1_2_BOF_Cum"] * df_p["BOF_LS_Input"]
    + df_p["Scope_1_2_Custom_Cum"] * df_p["Custom_Steel_LS_Input"]
)

df_p["Cast_Inputs_I_1"] = (
    df_p["Scope_1_EAF_Cum"] * df_p["EAF_LS_Input"]
    + df_p["Scope_1_Conarc_Cum"] * df_p["Conarc_LS_Input"]
    + df_p["Scope_1_BOF_Cum"] * df_p["BOF_LS_Input"]
    + df_p["Scope_1_Custom_Cum"] * df_p["Custom_Steel_LS_Input"]
)

df_p["Cast_Inputs_I_2"] = (
    (df_p["Scope_1_2_EAF_Cum"] - df_p["Scope_1_EAF_Cum"]) * df_p["EAF_LS_Input"]
    + (df_p["Scope_1_2_Conarc_Cum"] - df_p["Scope_1_Conarc_Cum"]) * df_p["Conarc_LS_Input"]
    + (df_p["Scope_1_2_BOF_Cum"] - df_p["Scope_1_BOF_Cum"]) * df_p["BOF_LS_Input"]
    + (df_p["Scope_1_2_Custom_Cum"] - df_p["Scope_1_Custom_Cum"]) * df_p["Custom_Steel_LS_Input"]
)

df_p["Cast_Up_I_1_2"] = df_p["Cast_I_1_2"] + df_p["Cast_Inputs_I_1_2"]
df_p["Cast_Up_I_1"] = df_p["Cast_1_I"] + df_p["Cast_Inputs_I_1"]
df_p["Cast_Up_I_2"] = df_p["Cast_2_I"] + df_p["Cast_Inputs_I_2"]
df_p["Treat_Up_I_1_2"] = df_p["Cast_Up_I_1_2"] + df_p["Treat_I_1_2"]
df_p["Treat_Up_I_1"] = df_p["Cast_Up_I_1"] + df_p["Treat_1_I"]
df_p["Treat_Up_I_2"] = df_p["Cast_Up_I_2"] + df_p["Treat_2_I"]

df_p = df_p.rename(columns={'Power': 'Power_I', 'Energy': 'Fuel_I'})

df_p["EAF?"] = df_p["Production_EAF"] > 0
df_p["BOF?"] = df_p["Production_BOF"] > 0
df_p["Both?"] = (df_p["Production_EAF"] > 0) & (df_p["Production_EAF"] > 0)

df_p['Carbon_Costs?'] = np.where(
    df_p["CO2_Costs"] > 0,
    'Face Carbon Costs',
    'Do Not Face Carbon Costs'
)

# ============================================================================
# [RECONSTRUCTED] Final schema of the 14 Oct output (cast_steel_clean.csv)
# ============================================================================
# Note: Power_total above is still summed per asset across all years (K14), so in this
# multi-year run caster and treatment Scope 2 are diluted roughly 19-fold. Kept so the
# output matches the 14 Oct file; fix by grouping on ["Asset", "Year"].
raw = pd.read_csv("Commodities/steel.csv")
raw = raw[raw['Operating status'] == 'Operating']
yrs = [c for c in raw.columns if c.isdigit() and int(c) <= 2024]
pm_vars = {'HFO power, MWh': 'hfo_power', 'Natural gas power, MWh': 'natural_gas_power',
           'Thermal coal power, MWh': 'coal_power', 'Aggregate/other renewables power, MWh': 'renewables_power'}
pm = raw[raw['Variable'].isin(pm_vars)].melt(id_vars=['Asset', 'Variable'], value_vars=yrs,
                                              var_name='Year', value_name='Value')
pm = (pm.groupby(['Asset', 'Year', 'Variable'])['Value'].first().unstack()
        .rename(columns=pm_vars).reset_index())
state = raw.groupby('Asset')['Asset - State'].first().fillna('Unknown').rename('State').reset_index()

df_p['Year'] = df_p['Year'].astype(str)
df_p = df_p.merge(state, on='Asset', how='left').merge(pm, on=['Asset', 'Year'], how='left')
for c in pm_vars.values():
    if c not in df_p.columns:
        df_p[c] = 0
    df_p[c] = df_p[c].fillna(0)

df_p = df_p.rename(columns={'Asset - Operator': 'Operator', 'Asset - Region': 'Region',
                            'Production_caster': 'production', 'Cast_Power_MW': 'Power_req',
                            'Cast_Up_I_1': 'scope_1_intensity', 'Cast_Up_I_2': 'scope_2_intensity',
                            'Cast_Up_I_1_2': 'scope_1_2_intensity'})
df_p['Energy_req'] = df_p['Fuel_I'] * df_p['production']
df_p['scope_1'] = df_p['scope_1_intensity'] * df_p['production']
df_p['scope_2'] = df_p['scope_2_intensity'] * df_p['production']
df_p['Year'] = df_p['Year'].astype(int)

final_cols = ['Commodity', 'Asset', 'Owners', 'Operator', 'Product', 'Operating status', 'Country',
              'Region', 'Year', 'employees', 'State', 'hfo_power', 'natural_gas_power', 'coal_power',
              'renewables_power', 'Fuel_I', 'Power_I', 'production', 'scope_1_2_intensity',
              'scope_1_intensity', 'scope_1', 'scope_2_intensity', 'scope_2', 'EAF?', 'BOF?', 'Both?',
              'Power_req', 'Energy_req']
df_p[final_cols].to_csv("cast_steel_clean.csv", index=False)
print("Processing complete. Output saved to cast_steel_clean.csv")
