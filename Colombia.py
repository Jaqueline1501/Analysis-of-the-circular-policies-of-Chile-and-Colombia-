# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 12:15:03 2025

@author: jaque
"""


#%% Importing MARIO 
import mario
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

#%%----------------------------------------------------------------------------#
# Part 1: Load and prepare EXIOBASE data
#----------------------------------------------------------------------------#
  
# Define file paths
IOT_PATH = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\IOT_2019_ixi_new'
EXTENSIONS_PATH = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\new_E_accounts.xlsx'
AGGREGATION_PATH = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\world_aggregated_CO.xlsx'


# Parse EXIOBASE Input-Output Tables
world = mario.parse_exiobase('IOT', 'Monetary', IOT_PATH)
 
#%%
# Check balancing of the model (should be balanced within 5% margin)

is_balanced = world.is_balanced('flows', data_set='baseline', margin=0.05, as_dataframe=False)
print(f"Model is balanced: {is_balanced}")
#%%
# Add GHG emissions and other environmental extensions
units = pd.read_excel(EXTENSIONS_PATH, sheet_name='units', index_col=[0], header=[0])
world.add_extensions(io=EXTENSIONS_PATH, units=units, matrix='E')

# Aggregate regions and sectors using predefined aggregation scheme
world.aggregate(
    io=AGGREGATION_PATH,
    levels=["Factor of production", "Satellite account", "Consumption category", "Region", "Sector"]
)
#%%
# Validate the aggregation by checking matrices
print("Available regions:", world.matrices['baseline']['E'].columns.unique())
print("Available environmental extensions:", world.matrices['baseline']['E'].index.unique())

#%%----------------------------------------------------------------------------#
# Part 2: Define key indicators for Triple Bottom Line (TBL) assessment
#----------------------------------------------------------------------------#

# Define employment indicators
EMPLOYMENT_INDICATORS = [
    'Employment people: Low-skilled male', 'Employment people: Low-skilled female',
    'Employment people: Medium-skilled male', 'Employment people: Medium-skilled female',
    'Employment people: High-skilled male', 'Employment people: High-skilled female'
]

# Define environmental and economic indicators
GHG_INDICATOR = 'GHG emissions (GWP100) from v3.8.2'
VALUE_ADDED_INDICATOR = 'Value Added'


#%%----------------------------------------------------------------------------#
# Part 3: Establish baseline scenario
#----------------------------------------------------------------------------#

# Extract baseline data from the model
baseline_data = world.query(matrices='F', scenarios='baseline')

# Extract baseline values for environmental indicators (global and LAC region)
baseline_ghg = baseline_data.loc[GHG_INDICATOR]
baseline_employment = baseline_data.loc[EMPLOYMENT_INDICATORS]

# Store baseline matrices for impact calculations
Y_baseline = world.Y  # Final demand
L_baseline = world.w  # Leontief inverse

# Extract key variables for footprint calculations
ghg_intensities = world.e.loc[GHG_INDICATOR, :]
ghg_lac_intensities = world.e.loc[GHG_INDICATOR, 'LAC']
value_added = world.v.loc[VALUE_ADDED_INDICATOR, :]

# Calculate baseline footprints (consumption-based impacts)
# Formula: Footprint = Diag(intensities) × Leontief × Final demand
ghg_footprint_baseline = np.diag(ghg_intensities) @ L_baseline
ghg_footprint_baseline_lac = ghg_footprint_baseline @ Y_baseline.loc[:, 'LAC']
ghg_footprint_baseline_lac_sum = ghg_footprint_baseline_lac.sum()

# Calculate baseline employment footprints
employment_baseline_sum = world.query(matrices='E', scenarios='baseline').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_baseline = np.diag(employment_baseline_sum) @ L_baseline
employment_footprint_baseline_lac = employment_footprint_baseline @ Y_baseline.loc[:, 'LAC']


# Calculate baseline value added footprints
value_added_footprint_baseline = np.diag(value_added) @ L_baseline
value_added_footprint_baseline_lac = value_added_footprint_baseline @ Y_baseline.loc[:, 'LAC']

#%%
emp = world.query(matrices='E', scenarios='baseline').loc[EMPLOYMENT_INDICATORS]
emp_lac = emp['LAC']


#%%----------------------------------------------------------------------------#
# Part 4: Modelling of the Individual scenarios 
#----------------------------------------------------------------------------#


#%%----------------------------------------------------------------------------#
# Part 4.1: Intervention 6: - Increasing Colombia's recycling rate to 17.9% from a baseline of 8.6% 
#----------------------------------------------------------------------------#

# Define scenario path
scenario6_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_CO\CE_CO_Intervention_6.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario6_path,
    z=True,
    Y=True,
    scenario='Scenario 6',
    force_rewrite=True,
    notes=['Increasing recycling in Colombia']
)

# Extract scenario data
scenario6_data = world.query(matrices='F', scenarios='Scenario 6')

# Store key matrices
Y_scenario6 = world.matrices['Scenario 6']['Y']
L_scenario6 = world.matrices['Scenario 6']['w']
e_scenario6 = world.matrices['Scenario 6']['e']

#%% Calculate GHG impacts
scenario6_ghg = scenario6_data.loc[GHG_INDICATOR]
ghg_comparison_s6 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 6 (Mt CO₂e)': scenario6_ghg / 1e6,
})
ghg_comparison_s6['Absolute Change (Mt CO₂e)'] = ghg_comparison_s6['Scenario 6 (Mt CO₂e)'] - ghg_comparison_s6['Baseline (Mt CO₂e)']
ghg_comparison_s6['Relative Change (%)'] = (ghg_comparison_s6['Absolute Change (Mt CO₂e)'] / ghg_comparison_s6['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario6:")
print(ghg_comparison_s6.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario6 = Y_scenario6.loc[:, 'LAC']
ghg_intensities_s6 = e_scenario6.loc[GHG_INDICATOR, :]
ghg_footprint_s6 = np.diag(ghg_intensities_s6) @ L_scenario6
ghg_footprint_s6_lac = ghg_footprint_s6 @ Y_lac_scenario6
ghg_footprint_s6_lac_sum = ghg_footprint_s6_lac.sum()


difference_abs_footprint_s6 = ghg_footprint_baseline_lac_sum - ghg_footprint_s6_lac_sum

# Calculate relative and absolute differences for GHG
ghg_diff_rel_s6 = (ghg_footprint_s6_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s6 = ghg_footprint_s6_lac.sum() - ghg_footprint_baseline_lac.sum()

# Worldwide changes 
GHG_s6 = world.query(
    matrices='E',
    scenarios='Scenario 6',
    ).loc[GHG_INDICATOR]

delta_ghg_s6_abs = world.query(
    matrices='E',
    scenarios='Scenario 6',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

delta_ghg_s6_rel = world.query(
    matrices='E',
    scenarios='Scenario 6',
    base_scenario='baseline',
    type='relative',
    ).loc[GHG_INDICATOR]


#%% Calculate employment impacts
scenario6_employment = scenario6_data.loc[EMPLOYMENT_INDICATORS]
employment_change_s6 = ((scenario6_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s6['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s6_sum = world.query(matrices='E', scenarios='Scenario 6').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_s6 = np.diag(employment_s6_sum) @ L_scenario6
employment_footprint_s6_lac = employment_footprint_s6 @ Y_scenario6.loc[:, 'LAC']

emp_diff_s6_sector = employment_footprint_baseline_lac - employment_footprint_s6_lac

emp_change_s6 = employment_footprint_s6_lac.sum() - employment_footprint_baseline_lac.sum()
emp_change_rel_s6 = emp_change_s6 / employment_footprint_baseline_lac.sum()

Emp_6 = world.query(
    matrices ='E',
    scenarios = 'Scenario 6',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s6 = world.query(
    matrices='E',
    scenarios='Scenario 6',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s6_rel = world.query(
    matrices='E',
    scenarios='Scenario 6',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s6 = world.query(matrices='E', scenarios='Scenario 6').loc[EMPLOYMENT_INDICATORS]
emp_lac_s6 = emp_s6['LAC']

#%% Calculate value added impacts
value_added_footprint_s6 = np.diag(value_added) @ L_scenario6
value_added_footprint_s6_lac = value_added_footprint_s6 @ Y_scenario6.loc[:, 'LAC']

va_diff_abs_s6 = value_added_footprint_s6_lac.sum() - value_added_footprint_baseline_lac.sum()
va_diff_rel_s6 = va_diff_abs_s6 / value_added_footprint_baseline_lac.sum()

value_added_sum_baseline = value_added_footprint_baseline_lac.sum()
value_added_sum_s6 = value_added_footprint_s6_lac.sum()


delta_VA_s6 = world.query(
    matrices="V",
    scenarios="Scenario 6",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]

#%%----------------------------------------------------------------------------#
# Part 4.2: Intervention 7 - Increasing renewable electricity production 
#----------------------------------------------------------------------------#

# Define scenario path
scenario7_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_CO\CE_CO_Intervention_7.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario7_path,  
    z=True,
    Y=True,
    scenario='Scenario 7',  
    force_rewrite=True,
    notes=['Increasing renewable energy production'] 
)

# Extract scenario data
scenario7_data = world.query(matrices='F', scenarios='Scenario 7')  

# Store key matrices
Y_scenario7 = world.matrices['Scenario 7']['Y']  
L_scenario7 = world.matrices['Scenario 7']['w']  
e_scenario7 = world.matrices['Scenario 7']['e'] 

#%% Calculate GHG impacts
scenario7_ghg = scenario7_data.loc[GHG_INDICATOR]
ghg_comparison_s7 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 7 (Mt CO₂e)': scenario7_ghg / 1e6,  
})
ghg_comparison_s7['Absolute Change (Mt CO₂e)'] = ghg_comparison_s7['Scenario 7 (Mt CO₂e)'] - ghg_comparison_s7['Baseline (Mt CO₂e)'] 
ghg_comparison_s7['Relative Change (%)'] = (ghg_comparison_s7['Absolute Change (Mt CO₂e)'] / ghg_comparison_s7['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 7:") 
print(ghg_comparison_s7.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario7 = Y_scenario7.loc[:, 'LAC']  
ghg_intensities_s7 = e_scenario7.loc[GHG_INDICATOR, :]  
ghg_footprint_s7 = np.diag(ghg_intensities_s7) @ L_scenario7  
ghg_footprint_s7_lac = ghg_footprint_s7 @ Y_lac_scenario7

ghg_footprint_7_lac_sum = ghg_footprint_s7_lac.sum()  

difference_abs_footprint_s7 = ghg_footprint_baseline_lac_sum - ghg_footprint_7_lac_sum  

# Calculate relative and absolute differences for GHG
ghg_diff_rel_s7 = (ghg_footprint_s7_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s7 = ghg_footprint_s7_lac.sum() - ghg_footprint_baseline_lac.sum()

# Worldwide changes 
GHG_s7 = world.query(
    matrices='E',
    scenarios='Scenario 7',
    ).loc[GHG_INDICATOR]


delta_ghg_s7_abs = world.query(
    matrices='E',
    scenarios='Scenario 7',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

delta_ghg_s7_rel = world.query(
    matrices='E',
    scenarios='Scenario 7',
    base_scenario='baseline',
    type='relative',
    ).loc[GHG_INDICATOR]


#%% Calculate employment impacts
scenario7_employment = scenario7_data.loc[EMPLOYMENT_INDICATORS]  
employment_change_s7 = ((scenario7_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s7['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s7_sum = world.query(matrices='E', scenarios='Scenario 7').loc[EMPLOYMENT_INDICATORS].sum()  
employment_footprint_s7 = np.diag(employment_s7_sum) @ L_scenario7  
employment_footprint_s7_lac = employment_footprint_s7 @ Y_scenario7.loc[:, 'LAC']  

emp_change_s7 = employment_footprint_s7_lac.sum() - employment_footprint_baseline_lac.sum() 
emp_change_rel_s7 = emp_change_s7 / employment_footprint_baseline_lac.sum()  


emp_change_s7_total_lac = employment_footprint_baseline_lac.sum() - employment_footprint_s7_lac.sum() 

Emp_7 = world.query(
    matrices ='E',
    scenarios = 'Scenario 7',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s7 = world.query(
    matrices='E',
    scenarios='Scenario 7',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s7_rel = world.query(
    matrices='E',
    scenarios='Scenario 7',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s7 = world.query(matrices='E', scenarios='Scenario 7').loc[EMPLOYMENT_INDICATORS]
emp_lac_s7 = emp_s7['LAC'] 


#%% Calculate value added impacts
value_added_footprint_s7 = np.diag(value_added) @ L_scenario7  
value_added_footprint_s7_lac = value_added_footprint_s7 @ Y_scenario7.loc[:, 'LAC']  

va_diff_abs_s7 = value_added_footprint_s7_lac.sum() - value_added_footprint_baseline_lac.sum()  
va_diff_rel_s7 = va_diff_abs_s7 / value_added_footprint_baseline_lac.sum()  

delta_VA_s7 = world.query(
    matrices="V",
    scenarios="Scenario 7",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]


#%%----------------------------------------------------------------------------#
# Part 4.3: Intervention 8 - Recovery and recycling or tires 
#----------------------------------------------------------------------------#

# Define scenario path
scenario8_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_CO\CE_CO_Intervention_8.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario8_path,  
    z=True,
    Y=True,
    scenario='Scenario 8', 
    force_rewrite=True,
    notes=['Recycling and recovery of Tires'] 
)

# Extract scenario data
scenario8_data = world.query(matrices='F', scenarios='Scenario 8')  

# Store key matrices
Y_scenario8 = world.matrices['Scenario 8']['Y']  
L_scenario8 = world.matrices['Scenario 8']['w']  
e_scenario8 = world.matrices['Scenario 8']['e']  

#%% Calculate GHG impacts
scenario8_ghg = scenario8_data.loc[GHG_INDICATOR]
ghg_comparison_s8 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 8 (Mt CO₂e)': scenario8_ghg / 1e6,  
})
ghg_comparison_s8['Absolute Change (Mt CO₂e)'] = ghg_comparison_s8['Scenario 8 (Mt CO₂e)'] - ghg_comparison_s8['Baseline (Mt CO₂e)'] 
ghg_comparison_s8['Relative Change (%)'] = (ghg_comparison_s8['Absolute Change (Mt CO₂e)'] / ghg_comparison_s8['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 8:")
print(ghg_comparison_s8.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario8 = Y_scenario8.loc[:, 'LAC']  
ghg_intensities_s8 = e_scenario8.loc[GHG_INDICATOR, :]  
ghg_footprint_s8 = np.diag(ghg_intensities_s8) @ L_scenario8  
ghg_footprint_s8_lac = ghg_footprint_s8 @ Y_lac_scenario8

ghg_footprint_8_lac_sum = ghg_footprint_s8_lac.sum() 

difference_abs_footprint_s8 = ghg_footprint_baseline_lac_sum - ghg_footprint_8_lac_sum  

# Calculate relative and absolute differences for GHG
ghg_diff_rel_s8 = (ghg_footprint_s8_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s8 = ghg_footprint_s8_lac.sum() - ghg_footprint_baseline_lac.sum()

# Worldwide changes 
GHG_s8 = world.query(
    matrices='E',
    scenarios='Scenario 8',
    ).loc[GHG_INDICATOR]


delta_ghg_s8_abs = world.query(
    matrices='E',
    scenarios='Scenario 8',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

delta_ghg_s8_rel = world.query(
    matrices='E',
    scenarios='Scenario 8',
    base_scenario='baseline',
    type='relative',
    ).loc[GHG_INDICATOR]


#%% Calculate employment impacts
scenario8_employment = scenario8_data.loc[EMPLOYMENT_INDICATORS] 
employment_change_s8 = ((scenario8_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s8['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s8_sum = world.query(matrices='E', scenarios='Scenario 8').loc[EMPLOYMENT_INDICATORS].sum() 
employment_footprint_s8 = np.diag(employment_s8_sum) @ L_scenario8  
employment_footprint_s8_lac = employment_footprint_s8 @ Y_scenario8.loc[:, 'LAC']  

emp_change_s8 = employment_footprint_s8_lac.sum() - employment_footprint_baseline_lac.sum()  
emp_change_rel_s8 = emp_change_s8 / employment_footprint_baseline_lac.sum()  


emp_change_s8_total_lac = employment_footprint_baseline_lac.sum() - employment_footprint_s8_lac.sum()  

Emp_8 = world.query(
    matrices ='E',
    scenarios = 'Scenario 8',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s8 = world.query(
    matrices='E',
    scenarios='Scenario 8',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s8_rel = world.query(
    matrices='E',
    scenarios='Scenario 8',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s8 = world.query(matrices='E', scenarios='Scenario 8').loc[EMPLOYMENT_INDICATORS]
emp_lac_s8 = emp_s8['LAC'] 


#%% Calculate value added impacts
value_added_footprint_s8 = np.diag(value_added) @ L_scenario8  
value_added_footprint_s8_lac = value_added_footprint_s8 @ Y_scenario8.loc[:, 'LAC']  

va_diff_abs_s8 = value_added_footprint_s8_lac.sum() - value_added_footprint_baseline_lac.sum() 
va_diff_rel_s8 = va_diff_abs_s8 / value_added_footprint_baseline_lac.sum()  


delta_VA_s8 = world.query(
    matrices="V",
    scenarios="Scenario 8",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]


#%%----------------------------------------------------------------------------#
# Part 4.4: Intervention 9 - Recycling of construction 
#----------------------------------------------------------------------------#

# Define scenario path
scenario9_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_CO\CE_CO_Intervention_9.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario9_path,
    z=True,
    Y=True,
    scenario='Scenario 9',
    force_rewrite=True,
    notes=['Recycling construction']
)

# Extract scenario data
scenario9_data = world.query(matrices='F', scenarios='Scenario 9')

# Store key matrices
Y_scenario9 = world.matrices['Scenario 9']['Y']
L_scenario9 = world.matrices['Scenario 9']['w']
e_scenario9 = world.matrices['Scenario 9']['e']

#%% Calculate GHG impacts
scenario9_ghg = scenario9_data.loc[GHG_INDICATOR]
ghg_comparison_s9 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 9 (Mt CO₂e)': scenario9_ghg / 1e6,
})
ghg_comparison_s9['Absolute Change (Mt CO₂e)'] = ghg_comparison_s9['Scenario 9 (Mt CO₂e)'] - ghg_comparison_s9['Baseline (Mt CO₂e)']
ghg_comparison_s9['Relative Change (%)'] = (ghg_comparison_s9['Absolute Change (Mt CO₂e)'] / ghg_comparison_s9['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 9:")
print(ghg_comparison_s9.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario9 = Y_scenario9.loc[:, 'LAC']
ghg_intensities_s9 = e_scenario9.loc[GHG_INDICATOR, :]
ghg_footprint_s9 = np.diag(ghg_intensities_s9) @ L_scenario9
ghg_footprint_s9_lac = ghg_footprint_s9 @ Y_lac_scenario9

ghg_footprint_9_lac_sum = ghg_footprint_s9_lac.sum()

difference_abs_footprint_s9 = ghg_footprint_baseline_lac_sum - ghg_footprint_9_lac_sum

# Calculate relative and absolute differences for GHG
ghg_diff_rel_s9 = (ghg_footprint_s9_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s9 = ghg_footprint_s9_lac.sum() - ghg_footprint_baseline_lac.sum()

# Worldwide changes 
GHG_s9 = world.query(
    matrices='E',
    scenarios='Scenario 9',
    ).loc[GHG_INDICATOR]


delta_ghg_s9_abs = world.query(
    matrices='E',
    scenarios='Scenario 9',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

delta_ghg_s9_rel = world.query(
    matrices='E',
    scenarios='Scenario 9',
    base_scenario='baseline',
    type='relative',
    ).loc[GHG_INDICATOR]


#%% Calculate employment impacts
scenario9_employment = scenario9_data.loc[EMPLOYMENT_INDICATORS]
employment_change_s9 = ((scenario9_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s9['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s9_sum = world.query(matrices='E', scenarios='Scenario 9').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_s9 = np.diag(employment_s9_sum) @ L_scenario9
employment_footprint_s9_lac = employment_footprint_s9 @ Y_scenario9.loc[:, 'LAC']

emp_change_s9 = employment_footprint_s9_lac.sum() - employment_footprint_baseline_lac.sum()
emp_change_rel_s9 = emp_change_s9 / employment_footprint_baseline_lac.sum()


emp_change_s9_total_lac = employment_footprint_baseline_lac.sum() - employment_footprint_s9_lac.sum()

Emp_9 = world.query(
    matrices ='E',
    scenarios = 'Scenario 9',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s9 = world.query(
    matrices='E',
    scenarios='Scenario 9',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s9_rel = world.query(
    matrices='E',
    scenarios='Scenario 9',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s9 = world.query(matrices='E', scenarios='Scenario 9').loc[EMPLOYMENT_INDICATORS]
emp_lac_s9 = emp_s9['LAC'] 



#%% Calculate value added impacts
value_added_footprint_s9 = np.diag(value_added) @ L_scenario9
value_added_footprint_s9_lac = value_added_footprint_s9 @ Y_scenario9.loc[:, 'LAC']

va_diff_abs_s9 = value_added_footprint_s9_lac.sum() - value_added_footprint_baseline_lac.sum()
va_diff_rel_s9 = va_diff_abs_s9 / value_added_footprint_baseline_lac.sum()

delta_VA_s9 = world.query(
    matrices="V",
    scenarios="Scenario 9",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]


#%%----------------------------------------------------------------------------#
# Part 4.5: Intervention 10 - Management of hazardous waste 
#----------------------------------------------------------------------------#

# Define scenario path
scenario10_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_CO\CE_CO_Intervention_10.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario10_path,
    z=True,
    Y=True,
    scenario='Scenario 10',
    force_rewrite=True,
    notes=['Management of hazardous waste']
)

# Extract scenario data
scenario10_data = world.query(matrices='F', scenarios='Scenario 10')

# Store key matrices
Y_scenario10 = world.matrices['Scenario 10']['Y']
L_scenario10 = world.matrices['Scenario 10']['w']
e_scenario10 = world.matrices['Scenario 10']['e']

#%% Calculate GHG impacts
scenario10_ghg = scenario10_data.loc[GHG_INDICATOR]
ghg_comparison_s10 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 10 (Mt CO₂e)': scenario10_ghg / 1e6,
})
ghg_comparison_s10['Absolute Change (Mt CO₂e)'] = ghg_comparison_s10['Scenario 10 (Mt CO₂e)'] - ghg_comparison_s10['Baseline (Mt CO₂e)']
ghg_comparison_s10['Relative Change (%)'] = (ghg_comparison_s10['Absolute Change (Mt CO₂e)'] / ghg_comparison_s10['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 10:")
print(ghg_comparison_s10.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario10 = Y_scenario10.loc[:, 'LAC']
ghg_intensities_s10 = e_scenario10.loc[GHG_INDICATOR, :]
ghg_footprint_s10 = np.diag(ghg_intensities_s10) @ L_scenario10
ghg_footprint_s10_lac = ghg_footprint_s10 @ Y_lac_scenario10

ghg_footprint_10_lac_sum = ghg_footprint_s10_lac.sum()

difference_abs_footprint_s10 = ghg_footprint_baseline_lac_sum - ghg_footprint_10_lac_sum

# Calculate relative and absolute differences for GHG
ghg_diff_rel_s10 = (ghg_footprint_s10_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s10 = ghg_footprint_s10_lac.sum() - ghg_footprint_baseline_lac.sum()


# Worldwide changes 
GHG_s10 = world.query(
    matrices='E',
    scenarios='Scenario 10',
    ).loc[GHG_INDICATOR]

delta_ghg_s10_abs = world.query(
    matrices='E',
    scenarios='Scenario 10',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

delta_ghg_s10_rel = world.query(
    matrices='E',
    scenarios='Scenario 10',
    base_scenario='baseline',
    type='relative',
    ).loc[GHG_INDICATOR]

#%% Calculate employment impacts
scenario10_employment = scenario10_data.loc[EMPLOYMENT_INDICATORS]
employment_change_s10 = ((scenario10_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s10['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s10_sum = world.query(matrices='E', scenarios='Scenario 10').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_s10 = np.diag(employment_s10_sum) @ L_scenario10
employment_footprint_s10_lac = employment_footprint_s10 @ Y_scenario10.loc[:, 'LAC']

emp_change_s10 = employment_footprint_s10_lac.sum() - employment_footprint_baseline_lac.sum()
emp_change_rel_s10 = emp_change_s10 / employment_footprint_baseline_lac.sum()


emp_change_s10_total_lac = employment_footprint_baseline_lac.sum() - employment_footprint_s10_lac.sum()

Emp_10 = world.query(
    matrices ='E',
    scenarios = 'Scenario 10',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s10 = world.query(
    matrices='E',
    scenarios='Scenario 10',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s10_rel = world.query(
    matrices='E',
    scenarios='Scenario 10',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s10 = world.query(matrices='E', scenarios='Scenario 10').loc[EMPLOYMENT_INDICATORS]
emp_lac_s10 = emp_s10['LAC'] 


#%% Calculate value added impacts
value_added_footprint_s10 = np.diag(value_added) @ L_scenario10
value_added_footprint_s10_lac = value_added_footprint_s10 @ Y_scenario10.loc[:, 'LAC']

va_diff_abs_s10 = value_added_footprint_s10_lac.sum() - value_added_footprint_baseline_lac.sum()
va_diff_rel_s10 = va_diff_abs_s10 / value_added_footprint_baseline_lac.sum()

delta_VA_s10 = world.query(
    matrices="V",
    scenarios="Scenario 10",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]


#%%----------------------------------------------------------------------------#
# Part 4.6: Intervention 11 - Waste package utilization
#----------------------------------------------------------------------------#

# Define scenario path
scenario11_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_CO\CE_CO_Intervention_11.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario11_path,
    z=True,
    Y=True,
    scenario='Scenario 11',
    force_rewrite=True,
    notes=['Waste package utilization']
)

# Extract scenario data
scenario11_data = world.query(matrices='F', scenarios='Scenario 11')

# Store key matrices
Y_scenario11 = world.matrices['Scenario 11']['Y']
L_scenario11 = world.matrices['Scenario 11']['w']
e_scenario11 = world.matrices['Scenario 11']['e']

#%% Calculate GHG impacts
scenario11_ghg = scenario11_data.loc[GHG_INDICATOR]
ghg_comparison_s11 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 11 (Mt CO₂e)': scenario11_ghg / 1e6,
})
ghg_comparison_s11['Absolute Change (Mt CO₂e)'] = ghg_comparison_s11['Scenario 11 (Mt CO₂e)'] - ghg_comparison_s11['Baseline (Mt CO₂e)']
ghg_comparison_s11['Relative Change (%)'] = (ghg_comparison_s11['Absolute Change (Mt CO₂e)'] / ghg_comparison_s11['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 11:")
print(ghg_comparison_s11.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario11 = Y_scenario11.loc[:, 'LAC']
ghg_intensities_s11 = e_scenario11.loc[GHG_INDICATOR, :]
ghg_footprint_s11 = np.diag(ghg_intensities_s11) @ L_scenario11
ghg_footprint_s11_lac = ghg_footprint_s11 @ Y_lac_scenario11

ghg_footprint_11_lac_sum = ghg_footprint_s11_lac.sum()

difference_abs_footprint_s11 = ghg_footprint_baseline_lac_sum - ghg_footprint_11_lac_sum

# Calculate relative and absolute differences for GHG
ghg_diff_rel_s11 = (ghg_footprint_s11_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s11 = ghg_footprint_s11_lac.sum() - ghg_footprint_baseline_lac.sum()


# Worldwide changes 
GHG_s11 = world.query(
    matrices='E',
    scenarios='Scenario 11',
    ).loc[GHG_INDICATOR]


delta_ghg_s11_abs = world.query(
    matrices='E',
    scenarios='Scenario 11',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

delta_ghg_s11_rel = world.query(
    matrices='E',
    scenarios='Scenario 11',
    base_scenario='baseline',
    type='relative',
    ).loc[GHG_INDICATOR]

#%% Calculate employment impacts
scenario11_employment = scenario11_data.loc[EMPLOYMENT_INDICATORS]
employment_change_s11 = ((scenario11_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s11['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s11_sum = world.query(matrices='E', scenarios='Scenario 11').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_s11 = np.diag(employment_s11_sum) @ L_scenario11
employment_footprint_s11_lac = employment_footprint_s11 @ Y_scenario11.loc[:, 'LAC']

emp_change_s11 = employment_footprint_s11_lac.sum() - employment_footprint_baseline_lac.sum()
emp_change_rel_s11 = emp_change_s11 / employment_footprint_baseline_lac.sum()

emp_change_s11_total_lac = employment_footprint_baseline_lac.sum() - employment_footprint_s11_lac.sum()

Emp_11 = world.query(
    matrices ='E',
    scenarios = 'Scenario 11',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s11 = world.query(
    matrices='E',
    scenarios='Scenario 11',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s11_rel = world.query(
    matrices='E',
    scenarios='Scenario 11',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s11 = world.query(matrices='E', scenarios='Scenario 11').loc[EMPLOYMENT_INDICATORS]
emp_lac_s11 = emp_s11['LAC'] 


#%% Calculate value added impacts
value_added_footprint_s11 = np.diag(value_added) @ L_scenario11
value_added_footprint_s11_lac = value_added_footprint_s11 @ Y_scenario11.loc[:, 'LAC']

va_diff_abs_s11 = value_added_footprint_s11_lac.sum() - value_added_footprint_baseline_lac.sum()
va_diff_rel_s11 = va_diff_abs_s11 / value_added_footprint_baseline_lac.sum()

delta_VA_s11 = world.query(
    matrices="V",
    scenarios="Scenario 11",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]


#%%----------------------------------------------------------------------------#
# Part 4.6: Intervention 12 - Metal scrap utilization
#----------------------------------------------------------------------------#


# Define scenario path
scenario12_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_CO\CE_CO_Intervention_12.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario12_path,
    z=True,
    scenario='Scenario 12',
    force_rewrite=True,
    notes=['Metal scrap utilization']
)

# Extract scenario data
scenario12_data = world.query(matrices='F', scenarios='Scenario 12')

# Store key matrices
Y_scenario12 = world.matrices['Scenario 12']['Y']
L_scenario12 = world.matrices['Scenario 12']['w']
e_scenario12 = world.matrices['Scenario 12']['e']

#%% Calculate GHG impacts
scenario12_ghg = scenario12_data.loc[GHG_INDICATOR]
ghg_comparison_s12 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 12 (Mt CO₂e)': scenario12_ghg / 1e6,
})
ghg_comparison_s12['Absolute Change (Mt CO₂e)'] = ghg_comparison_s12['Scenario 12 (Mt CO₂e)'] - ghg_comparison_s12['Baseline (Mt CO₂e)']
ghg_comparison_s12['Relative Change (%)'] = (ghg_comparison_s12['Absolute Change (Mt CO₂e)'] / ghg_comparison_s12['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 12:")
print(ghg_comparison_s12.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario12 = Y_scenario12.loc[:, 'LAC']
ghg_intensities_s12 = e_scenario12.loc[GHG_INDICATOR, :]
ghg_footprint_s12 = np.diag(ghg_intensities_s12) @ L_scenario12
ghg_footprint_s12_lac = ghg_footprint_s12 @ Y_lac_scenario12

ghg_footprint_12_lac_sum = ghg_footprint_s12_lac.sum()

difference_abs_footprint_s12 = ghg_footprint_baseline_lac_sum - ghg_footprint_12_lac_sum

# Calculate relative and absolute differences for GHG
ghg_diff_rel_s12 = (ghg_footprint_s12_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s12 = ghg_footprint_s12_lac.sum() - ghg_footprint_baseline_lac.sum()

# Worldwide changes 
GHG_s12 = world.query(
    matrices='E',
    scenarios='Scenario 12',
    ).loc[GHG_INDICATOR]


delta_ghg_s12_abs = world.query(
    matrices='E',
    scenarios='Scenario 12',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

delta_ghg_s12_rel = world.query(
    matrices='E',
    scenarios='Scenario 12',
    base_scenario='baseline',
    type='relative',
    ).loc[GHG_INDICATOR]


#%% Calculate employment impacts
scenario12_employment = scenario12_data.loc[EMPLOYMENT_INDICATORS]
employment_change_s12 = ((scenario12_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s12['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s12_sum = world.query(matrices='E', scenarios='Scenario 12').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_s12 = np.diag(employment_s12_sum) @ L_scenario12
employment_footprint_s12_lac = employment_footprint_s12 @ Y_scenario12.loc[:, 'LAC']

emp_change_s12 = employment_footprint_s12_lac.sum() - employment_footprint_baseline_lac.sum()
emp_change_rel_s12 = emp_change_s12 / employment_footprint_baseline_lac.sum()


emp_change_s12_total_lac = employment_footprint_baseline_lac.sum() - employment_footprint_s12_lac.sum()

Emp_12 = world.query(
    matrices ='E',
    scenarios = 'Scenario 12',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s12 = world.query(
    matrices='E',
    scenarios='Scenario 12',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s12_rel = world.query(
    matrices='E',
    scenarios='Scenario 12',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s12 = world.query(matrices='E', scenarios='Scenario 12').loc[EMPLOYMENT_INDICATORS]
emp_lac_s12 = emp_s12['LAC'] 

#%% Calculate value added impacts
value_added_footprint_s12 = np.diag(value_added) @ L_scenario12
value_added_footprint_s12_lac = value_added_footprint_s12 @ Y_scenario12.loc[:, 'LAC']

va_diff_abs_s12 = value_added_footprint_s12_lac.sum() - value_added_footprint_baseline_lac.sum()
va_diff_rel_s12 = va_diff_abs_s12 / value_added_footprint_baseline_lac.sum()

delta_VA_s12 = world.query(
    matrices="V",
    scenarios="Scenario 12",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]


#%%----------------------------------------------------------------------------#
# Part 4.7: Intervention 13 - Biogas generation
#----------------------------------------------------------------------------#

# Define scenario path
scenario13_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_CO\CE_CO_Intervention_13.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario13_path,
    z=True,
    Y=True,
    scenario='Scenario 13',
    force_rewrite=True,
    notes=['Biogas generation']
)

# Extract scenario data
scenario13_data = world.query(matrices='F', scenarios='Scenario 13')

# Store key matrices
Y_scenario13 = world.matrices['Scenario 13']['Y']
L_scenario13 = world.matrices['Scenario 13']['w']
e_scenario13 = world.matrices['Scenario 13']['e']


#%% Calculate GHG impacts
scenario13_ghg = scenario13_data.loc[GHG_INDICATOR]
ghg_comparison_s13 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 13 (Mt CO₂e)': scenario13_ghg / 1e6,
})
ghg_comparison_s13['Absolute Change (Mt CO₂e)'] = ghg_comparison_s13['Scenario 13 (Mt CO₂e)'] - ghg_comparison_s13['Baseline (Mt CO₂e)']
ghg_comparison_s13['Relative Change (%)'] = (ghg_comparison_s13['Absolute Change (Mt CO₂e)'] / ghg_comparison_s13['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 13:")
print(ghg_comparison_s13.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario13 = Y_scenario13.loc[:, 'LAC']
ghg_intensities_s13 = e_scenario13.loc[GHG_INDICATOR, :]
ghg_footprint_s13 = np.diag(ghg_intensities_s13) @ L_scenario13
ghg_footprint_s13_lac = ghg_footprint_s13 @ Y_lac_scenario13

ghg_footprint_13_lac_sum = ghg_footprint_s13_lac.sum()

difference_abs_footprint_s13 = ghg_footprint_baseline_lac_sum - ghg_footprint_13_lac_sum

# Calculate relative and absolute differences for GHG
ghg_diff_rel_s13 = (ghg_footprint_s13_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s13 = ghg_footprint_s13_lac.sum() - ghg_footprint_baseline_lac.sum()


# Worldwide changes 
GHG_s13 = world.query(
    matrices='E',
    scenarios='Scenario 13',
    ).loc[GHG_INDICATOR]


delta_ghg_s13_abs = world.query(
    matrices='E',
    scenarios='Scenario 13',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

delta_ghg_s13_rel = world.query(
    matrices='E',
    scenarios='Scenario 13',
    base_scenario='baseline',
    type='relative',
    ).loc[GHG_INDICATOR]

#%% Calculate employment impacts
scenario13_employment = scenario13_data.loc[EMPLOYMENT_INDICATORS]
employment_change_s13 = ((scenario13_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s13['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s13_sum = world.query(matrices='E', scenarios='Scenario 13').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_s13 = np.diag(employment_s13_sum) @ L_scenario13
employment_footprint_s13_lac = employment_footprint_s13 @ Y_scenario13.loc[:, 'LAC']

emp_change_s13 = employment_footprint_s13_lac.sum() - employment_footprint_baseline_lac.sum()
emp_change_rel_s13 = emp_change_s13 / employment_footprint_baseline_lac.sum()


emp_change_s13_total_lac = employment_footprint_baseline_lac.sum() - employment_footprint_s13_lac.sum()

Emp_13 = world.query(
    matrices ='E',
    scenarios = 'Scenario 13',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s13 = world.query(
    matrices='E',
    scenarios='Scenario 13',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s13_rel = world.query(
    matrices='E',
    scenarios='Scenario 13',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s13 = world.query(matrices='E', scenarios='Scenario 13').loc[EMPLOYMENT_INDICATORS]
emp_lac_s13 = emp_s13['LAC'] 

#%% Calculate value added impacts
value_added_footprint_s13 = np.diag(value_added) @ L_scenario13
value_added_footprint_s13_lac = value_added_footprint_s13 @ Y_scenario13.loc[:, 'LAC']

va_diff_abs_s13 = value_added_footprint_s13_lac.sum() - value_added_footprint_baseline_lac.sum()
va_diff_rel_s13 = va_diff_abs_s13 / value_added_footprint_baseline_lac.sum()

delta_VA_s13 = world.query(
    matrices="V",
    scenarios="Scenario 13",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]



