# -*- coding: utf-8 -*-
"""
Created on Thu May  8 09:20:54 2025

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
AGGREGATION_PATH = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\world_aggregated_CH.xlsx'


# Parse EXIOBASE Input-Output Tables
world = mario.parse_exiobase('IOT', 'Monetary', IOT_PATH)
 

# Check balancing of the model (should be balanced within 5% margin)
is_balanced = world.is_balanced('flows', data_set='baseline', margin=0.05, as_dataframe=False)
print(f"Model is balanced: {is_balanced}")

# Add GHG emissions and other environmental extensions
units = pd.read_excel(EXTENSIONS_PATH, sheet_name='units', index_col=[0], header=[0])
world.add_extensions(io=EXTENSIONS_PATH, units=units, matrix='E')

# Aggregate regions and sectors using predefined aggregation scheme
world.aggregate(
    io=AGGREGATION_PATH,
    levels=["Factor of production", "Satellite account", "Consumption category", "Region", "Sector"]
)

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




#%%----------------------------------------------------------------------------#
# Part 4: Modelling of the Individual scenarios 
#----------------------------------------------------------------------------#


#%%----------------------------------------------------------------------------#
# Part 4.1: Intervention 1: - Reducing MSW Waste in Chile 
#----------------------------------------------------------------------------#

# Define scenario path
scenario1_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_Chile\CE_CH_Scenario_1.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario1_path,
    z=True,
    Y=True,
    scenario='Scenario 1',
    force_rewrite=True,
    notes=['Reducing MSW waste in Chile']
)

# Extract scenario data
scenario1_data = world.query(matrices='F', scenarios='Scenario 1')

# Store key matrices
Y_scenario1 = world.matrices['Scenario 1']['Y']
L_scenario1 = world.matrices['Scenario 1']['w']
e_scenario1 = world.matrices['Scenario 1']['e']

#%% Calculate GHG impacts
scenario1_ghg = scenario1_data.loc[GHG_INDICATOR]
ghg_comparison_s1 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 1 (Mt CO₂e)': scenario1_ghg / 1e6,
})
ghg_comparison_s1['Absolute Change (Mt CO₂e)'] = ghg_comparison_s1['Scenario 1 (Mt CO₂e)'] - ghg_comparison_s1['Baseline (Mt CO₂e)']
ghg_comparison_s1['Relative Change (%)'] = (ghg_comparison_s1['Absolute Change (Mt CO₂e)'] / ghg_comparison_s1['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 1:")
print(ghg_comparison_s1.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario1 = Y_scenario1.loc[:, 'LAC']
ghg_intensities_s1 = e_scenario1.loc[GHG_INDICATOR, :]
ghg_footprint_s1 = np.diag(ghg_intensities_s1) @ L_scenario1
ghg_footprint_s1_lac = ghg_footprint_s1 @ Y_lac_scenario1
ghg_footprint_s1_lac_sum = ghg_footprint_s1_lac.sum()


difference_abs_footprint_s1 = ghg_footprint_baseline_lac_sum - ghg_footprint_s1_lac_sum


# Calculate relative and absolute differences for GHG
ghg_diff_rel_s1 = (ghg_footprint_s1_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s1 = ghg_footprint_s1_lac.sum() - ghg_footprint_baseline_lac.sum()


# Worldwide changes 
GHG_s1 = world.query(
    matrices='E',
    scenarios='Scenario 1',
    ).loc[GHG_INDICATOR]


delta_ghg_s1_abs = world.query(
    matrices='E',
    scenarios='Scenario 1',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

delta_ghg_s1_rel = world.query(
    matrices='E',
    scenarios='Scenario 1',
    base_scenario='baseline',
    type='relative',
    ).loc[GHG_INDICATOR]

#%%
emp = world.query(matrices='E', scenarios='baseline').loc[EMPLOYMENT_INDICATORS]
emp_lac = emp['LAC']


#%% Calculate employment impacts

scenario1_employment = scenario1_data.loc[EMPLOYMENT_INDICATORS]
employment_change_s1 = ((scenario1_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s1['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s1_sum = world.query(matrices='E', scenarios='Scenario 1').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_s1 = np.diag(employment_s1_sum) @ L_scenario1
employment_footprint_s1_lac = employment_footprint_s1 @ Y_scenario1.loc[:, 'LAC']

emp_diff_s1_sector = employment_footprint_baseline_lac - employment_footprint_s1_lac

emp_change_s1 = employment_footprint_s1_lac.sum() - employment_footprint_baseline_lac.sum()
emp_change_rel_s1 = emp_change_s1 / employment_footprint_baseline_lac.sum()


Emp_1 = world.query(
    matrices ='E',
    scenarios = 'Scenario 1',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s1 = world.query(
    matrices='E',
    scenarios='Scenario 1',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s1_rel = world.query(
    matrices='E',
    scenarios='Scenario 1',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s1 = world.query(matrices='E', scenarios='Scenario 1').loc[EMPLOYMENT_INDICATORS]
emp_lac_s1 = emp_s1['LAC']


#%% Calculate value added impacts
value_added_footprint_s1 = np.diag(value_added) @ L_scenario1
value_added_footprint_s1_lac = value_added_footprint_s1 @ Y_scenario1.loc[:, 'LAC']

va_diff_abs_s1 = value_added_footprint_s1_lac.sum() - value_added_footprint_baseline_lac.sum()
va_diff_rel_s1 = va_diff_abs_s1 / value_added_footprint_baseline_lac.sum()

value_added_sum_baseline = value_added_footprint_baseline_lac.sum()
value_added_sum_s1 = value_added_footprint_s1_lac.sum()


delta_VA_s1 = world.query(
    matrices="V",
    scenarios="Scenario 1",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]

#%%----------------------------------------------------------------------------#
# Part 4.2: Scenario 2 - Reducing total waste generated by 30% until 2040
#----------------------------------------------------------------------------#

# Define scenario path
scenario2_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_Chile\CE_CH_Scenario_2.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario2_path,
    z=True,
    Y=True,
    scenario='Scenario 2',
    force_rewrite=True,
    notes=['Reducing total waste generated by 30% until 2040']
)

# Extract scenario data
scenario2_data = world.query(matrices='F', scenarios='Scenario 2')

# Store key matrices
Y_scenario2 = world.matrices['Scenario 2']['Y']
L_scenario2 = world.matrices['Scenario 2']['w']
e_scenario2 = world.matrices['Scenario 2']['e']

#%% Calculate GHG impacts
scenario2_ghg = scenario2_data.loc[GHG_INDICATOR]
ghg_comparison_s2 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 2 (Mt CO₂e)': scenario2_ghg / 1e6,
})
ghg_comparison_s2['Absolute Change (Mt CO₂e)'] = ghg_comparison_s2['Scenario 2 (Mt CO₂e)'] - ghg_comparison_s2['Baseline (Mt CO₂e)']
ghg_comparison_s2['Relative Change (%)'] = (ghg_comparison_s2['Absolute Change (Mt CO₂e)'] / ghg_comparison_s2['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 2:")
print(ghg_comparison_s2.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario2 = Y_scenario2.loc[:, 'LAC']
ghg_intensities_s2 = e_scenario2.loc[GHG_INDICATOR, :]
ghg_footprint_s2 = np.diag(ghg_intensities_s2) @ L_scenario2
ghg_footprint_s2_lac = ghg_footprint_s2 @ Y_lac_scenario2

ghg_footprint_2_lac_sum = ghg_footprint_s2_lac.sum()

difference_abs_footprint_s2 = ghg_footprint_baseline_lac_sum - ghg_footprint_2_lac_sum

delta_E_s2 = world.query(
    matrices='E',
    scenarios='Scenario 2',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

# Calculate relative and absolute differences for GHG
ghg_diff_rel_s2 = (ghg_footprint_s2_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s2 = ghg_footprint_s2_lac.sum() - ghg_footprint_baseline_lac.sum()



#%% Calculate employment impacts
scenario2_employment = scenario2_data.loc[EMPLOYMENT_INDICATORS]
employment_change_s2 = ((scenario2_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s2['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s2_sum = world.query(matrices='E', scenarios='Scenario 2').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_s2 = np.diag(employment_s2_sum) @ L_scenario2
employment_footprint_s2_lac = employment_footprint_s2 @ Y_scenario2.loc[:, 'LAC']

# Calculate employment delta
employment_delta_s2 = world.query(
    matrices='E',
    scenarios='Scenario 2',
    base_scenario='baseline',
    type='absolute'
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s2_rel = world.query(
    matrices='E',
    scenarios='Scenario 2',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()


emp_change_s2 = employment_footprint_s2_lac.sum() - employment_footprint_baseline_lac.sum()
emp_change_rel_s2 = emp_change_s2 / employment_footprint_baseline_lac.sum()


emp_change_s2_total_lac = employment_footprint_baseline_lac.sum() - employment_footprint_s2_lac.sum()

Emp_2 = world.query(
    matrices ='E',
    scenarios = 'Scenario 2',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s2 = world.query(
    matrices='E',
    scenarios='Scenario 2',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s2_rel = world.query(
    matrices='E',
    scenarios='Scenario 2',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s2 = world.query(matrices='E', scenarios='Scenario 2').loc[EMPLOYMENT_INDICATORS]
emp_lac_s2 = emp_s2['LAC']


#%% Calculate value added impacts
value_added_footprint_s2 = np.diag(value_added) @ L_scenario2
value_added_footprint_s2_lac = value_added_footprint_s2 @ Y_scenario2.loc[:, 'LAC']

va_diff_abs_s2 = value_added_footprint_s2_lac.sum() - value_added_footprint_baseline_lac.sum()
va_diff_rel_s2 = va_diff_abs_s2 / value_added_footprint_baseline_lac.sum()

delta_VA_s2 = world.query(
    matrices="V",
    scenarios="Scenario 2",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]



#%%----------------------------------------------------------------------------#
# Part 4.3: Scenario 3 - Increasing general recycling rate
#----------------------------------------------------------------------------#

# Define scenario path
scenario3_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_Chile\CE_CH_Scenario_3.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario3_path,
    z=True,
    scenario='Scenario 3',
    force_rewrite=True,
    notes=['Increasing general recycling rate']
)

# Extract scenario data
scenario3_data = world.query(matrices='F', scenarios='Scenario 3')

# Store key matrices
Y_scenario3 = world.matrices['Scenario 3']['Y']
L_scenario3 = world.matrices['Scenario 3']['w']
e_scenario3 = world.matrices['Scenario 3']['e']

#%% Calculate GHG impacts
scenario3_ghg = scenario3_data.loc[GHG_INDICATOR]
ghg_comparison_s3 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 3 (Mt CO₂e)': scenario3_ghg / 1e6,
})
ghg_comparison_s3['Absolute Change (Mt CO₂e)'] = ghg_comparison_s3['Scenario 3 (Mt CO₂e)'] - ghg_comparison_s3['Baseline (Mt CO₂e)']
ghg_comparison_s3['Relative Change (%)'] = (ghg_comparison_s3['Absolute Change (Mt CO₂e)'] / ghg_comparison_s3['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 3:")
print(ghg_comparison_s3.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario3 = Y_scenario3.loc[:, 'LAC']
ghg_intensities_s3 = e_scenario3.loc[GHG_INDICATOR, :]
ghg_footprint_s3 = np.diag(ghg_intensities_s3) @ L_scenario3
ghg_footprint_s3_lac = ghg_footprint_s3 @ Y_lac_scenario3

ghg_footprint_3_lac_sum = ghg_footprint_s3_lac.sum()

difference_abs_footprint_s3 = ghg_footprint_baseline_lac_sum - ghg_footprint_3_lac_sum

delta_E_s3 = world.query(
    matrices='E',
    scenarios='Scenario 3',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

# Calculate relative and absolute differences for GHG
ghg_diff_rel_s3 = (ghg_footprint_s3_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s3 = ghg_footprint_s3_lac.sum() - ghg_footprint_baseline_lac.sum()



#%% Calculate employment impacts
scenario3_employment = scenario3_data.loc[EMPLOYMENT_INDICATORS]
employment_change_s3 = ((scenario3_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s3['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s3_sum = world.query(matrices='E', scenarios='Scenario 3').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_s3 = np.diag(employment_s3_sum) @ L_scenario3
employment_footprint_s3_lac = employment_footprint_s3 @ Y_scenario3.loc[:, 'LAC']

# Calculate employment delta
employment_delta_s3 = world.query(
    matrices='E',
    scenarios='Scenario 3',
    base_scenario='baseline',
    type='absolute'
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s3_rel = world.query(
    matrices='E',
    scenarios='Scenario 3',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_change_s3 = employment_footprint_s3_lac.sum() - employment_footprint_baseline_lac.sum()
emp_change_rel_s3 = emp_change_s3 / employment_footprint_baseline_lac.sum()

emp_change_s3_total_lac = employment_footprint_baseline_lac.sum() - employment_footprint_s3_lac.sum()


Emp_3 = world.query(
    matrices ='E',
    scenarios = 'Scenario 3',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s3 = world.query(
    matrices='E',
    scenarios='Scenario 3',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s3_rel = world.query(
    matrices='E',
    scenarios='Scenario 3',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s3 = world.query(matrices='E', scenarios='Scenario 3').loc[EMPLOYMENT_INDICATORS]
emp_lac_s3 = emp_s3['LAC']



#%% Calculate value added impacts
value_added_footprint_s3 = np.diag(value_added) @ L_scenario3
value_added_footprint_s3_lac = value_added_footprint_s3 @ Y_scenario3.loc[:, 'LAC']

va_diff_abs_s3 = value_added_footprint_s3_lac.sum() - value_added_footprint_baseline_lac.sum()
va_diff_rel_s3 = va_diff_abs_s3 / value_added_footprint_baseline_lac.sum()

value_added_sum_s3 = value_added_footprint_s3_lac.sum()

delta_VA_s3 = world.query(
    matrices="V",
    scenarios="Scenario 3",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]


#%%----------------------------------------------------------------------------#
# Part 4.4: Scenario 4 - Increased recycling of MSW waste
#----------------------------------------------------------------------------#

# Define scenario path
scenario4_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_Chile\CE_CH_Scenario_4.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario4_path,
    z=True,
    Y=True,
    scenario='Scenario 4',
    force_rewrite=True,
    notes=['Increased recycling of MSW Waste']
)

# Extract scenario data
scenario4_data = world.query(matrices='F', scenarios='Scenario 4')

# Store key matrices
Y_scenario4 = world.matrices['Scenario 4']['Y']
L_scenario4 = world.matrices['Scenario 4']['w']
e_scenario4 = world.matrices['Scenario 4']['e']

#%% Calculate GHG impacts
scenario4_ghg = scenario4_data.loc[GHG_INDICATOR]
ghg_comparison_s4 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 4 (Mt CO₂e)': scenario4_ghg / 1e6,
})
ghg_comparison_s4['Absolute Change (Mt CO₂e)'] = ghg_comparison_s4['Scenario 4 (Mt CO₂e)'] - ghg_comparison_s4['Baseline (Mt CO₂e)']
ghg_comparison_s4['Relative Change (%)'] = (ghg_comparison_s4['Absolute Change (Mt CO₂e)'] / ghg_comparison_s4['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 4:")
print(ghg_comparison_s4.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario4 = Y_scenario4.loc[:, 'LAC']
ghg_intensities_s4 = e_scenario4.loc[GHG_INDICATOR, :]
ghg_footprint_s4 = np.diag(ghg_intensities_s4) @ L_scenario4
ghg_footprint_s4_lac = ghg_footprint_s4 @ Y_lac_scenario4

ghg_footprint_4_lac_sum = ghg_footprint_s4_lac.sum()

difference_abs_footprint_s4 = ghg_footprint_baseline_lac_sum - ghg_footprint_4_lac_sum

delta_E_s4 = world.query(
    matrices='E',
    scenarios='Scenario 4',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]

# Calculate relative and absolute differences for GHG
ghg_diff_rel_s4 = (ghg_footprint_s4_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s4 = ghg_footprint_s4_lac.sum() - ghg_footprint_baseline_lac.sum()



#%% Calculate employment impacts
scenario4_employment = scenario4_data.loc[EMPLOYMENT_INDICATORS]
employment_change_s4 = ((scenario4_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s4['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s4_sum = world.query(matrices='E', scenarios='Scenario 4').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_s4 = np.diag(employment_s4_sum) @ L_scenario4
employment_footprint_s4_lac = employment_footprint_s4 @ Y_scenario4.loc[:, 'LAC']

# Calculate employment delta
employment_delta_s4 = world.query(
    matrices='E',
    scenarios='Scenario 4',
    base_scenario='baseline',
    type='absolute'
).loc[EMPLOYMENT_INDICATORS].sum()

employment_delta_4_rel = world.query(
    matrices='E',
    scenarios='Scenario 4',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_change_s4 = employment_footprint_s4_lac.sum() - employment_footprint_baseline_lac.sum()
emp_change_rel_s4 = emp_change_s4 / employment_footprint_baseline_lac.sum()

emp_change_s4_total_lac = employment_footprint_baseline_lac.sum() - employment_footprint_s4_lac.sum()


Emp_4 = world.query(
    matrices ='E',
    scenarios = 'Scenario 4',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s4 = world.query(
    matrices='E',
    scenarios='Scenario 4',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s4_rel = world.query(
    matrices='E',
    scenarios='Scenario 4',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s4 = world.query(matrices='E', scenarios='Scenario 4').loc[EMPLOYMENT_INDICATORS]
emp_lac_s4 = emp_s4['LAC']


#%% Calculate value added impacts
value_added_footprint_s4 = np.diag(value_added) @ L_scenario4
value_added_footprint_s4_lac = value_added_footprint_s4 @ Y_scenario4.loc[:, 'LAC']

va_diff_abs_s4 = value_added_footprint_s4_lac.sum() - value_added_footprint_baseline_lac.sum()
va_diff_rel_s4 = va_diff_abs_s4 / value_added_footprint_baseline_lac.sum()


value_added_sum_s4 = value_added_footprint_s4_lac.sum()


delta_VA_s4 = world.query(
    matrices="V",
    scenarios="Scenario 4",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]

#%%----------------------------------------------------------------------------#
# Part 4.5: Scenario 5 - Increasing waste collected and treated
#----------------------------------------------------------------------------#

# Define scenario path
scenario5_path = r'C:\Users\jaque\Downloads\IE -Y2\Thesis_midterm\Scenarios_Chile\CE_CH_Intervention_5.xlsx'

# Apply shock to the economic system
world.shock_calc(
    io=scenario5_path,
    z=True,
    scenario='Scenario 5',
    force_rewrite=True,
    notes=['Increasing waste collected and treated']
)

# Extract scenario data
scenario5_data = world.query(matrices='F', scenarios='Scenario 5')

# Store key matrices
Y_scenario5 = world.matrices['Scenario 5']['Y']
L_scenario5 = world.matrices['Scenario 5']['w']
e_scenario5 = world.matrices['Scenario 5']['e']

#%% Calculate GHG impacts
scenario5_ghg = scenario5_data.loc[GHG_INDICATOR]
ghg_comparison_s5 = pd.DataFrame({
    'Baseline (Mt CO₂e)': baseline_ghg / 1e6,
    'Scenario 5 (Mt CO₂e)': scenario5_ghg / 1e6,
})
ghg_comparison_s5['Absolute Change (Mt CO₂e)'] = ghg_comparison_s5['Scenario 5 (Mt CO₂e)'] - ghg_comparison_s5['Baseline (Mt CO₂e)']
ghg_comparison_s5['Relative Change (%)'] = (ghg_comparison_s5['Absolute Change (Mt CO₂e)'] / ghg_comparison_s5['Baseline (Mt CO₂e)']) * 100

print("\nGHG Comparison for Scenario 5:")
print(ghg_comparison_s5.round(2))

# Calculate LAC-specific GHG footprints
Y_lac_scenario5 = Y_scenario5.loc[:, 'LAC']
ghg_intensities_s5 = e_scenario5.loc[GHG_INDICATOR, :]
ghg_footprint_s5 = np.diag(ghg_intensities_s5) @ L_scenario5
ghg_footprint_s5_lac = ghg_footprint_s5 @ Y_lac_scenario5

delta_E_s5 = world.query(
    matrices='E',
    scenarios='Scenario 5',
    base_scenario='baseline',
    type='absolute',
    ).loc[GHG_INDICATOR]


delta_E_s5_rel = world.query(
    matrices='E',
    scenarios='Scenario 5',
    base_scenario='baseline',
    type='relative',
    ).loc[GHG_INDICATOR]

ghg_s5 = world.query(matrices='E', scenarios='Scenario 5').loc[GHG_INDICATOR]
ghg_lac_s5 = ghg_s5['LAC']


# Calculate relative and absolute differences for GHG
ghg_diff_rel_s5 = (ghg_footprint_s5_lac.sum() - ghg_footprint_baseline_lac.sum()) / ghg_footprint_baseline_lac.sum()
ghg_diff_abs_s5 = ghg_footprint_s5_lac.sum() - ghg_footprint_baseline_lac.sum()

ghg_footprint_5_lac_sum = ghg_footprint_s5_lac.sum()

difference_abs_footprint_s5 = ghg_footprint_baseline_lac_sum - ghg_footprint_5_lac_sum


#%% Calculate employment impacts
scenario5_employment = scenario5_data.loc[EMPLOYMENT_INDICATORS]
employment_change_s5 = ((scenario5_employment - baseline_employment) / baseline_employment) * 100

print("\nRelative Change in Employment Indicators for LAC (in %):")
print(employment_change_s5['LAC'].round(2))

# Calculate LAC-specific employment footprints
employment_s5_sum = world.query(matrices='E', scenarios='Scenario 5').loc[EMPLOYMENT_INDICATORS].sum()
employment_footprint_s5 = np.diag(employment_s5_sum) @ L_scenario5
employment_footprint_s5_lac = employment_footprint_s5 @ Y_scenario5.loc[:, 'LAC']

emp_change_s5 = employment_footprint_s5_lac.sum() - employment_footprint_baseline_lac.sum()
emp_change_rel_s5 = emp_change_s5 / employment_footprint_baseline_lac.sum()

emp_change_s5_total_lac = employment_footprint_baseline_lac.sum() - employment_footprint_s5_lac.sum()


Emp_5 = world.query(
    matrices ='E',
    scenarios = 'Scenario 5',
    ).loc[EMPLOYMENT_INDICATORS]

# Calculate employment delta
employment_delta_s5 = world.query(
    matrices='E',
    scenarios='Scenario 5',
    base_scenario='baseline',
    type='absolute',
).loc[EMPLOYMENT_INDICATORS].sum()

# Calculate employment delta
employment_delta_s5_rel = world.query(
    matrices='E',
    scenarios='Scenario 5',
    base_scenario='baseline',
    type='relative'
).loc[EMPLOYMENT_INDICATORS].sum()

emp_s5 = world.query(matrices='E', scenarios='Scenario 5').loc[EMPLOYMENT_INDICATORS]
emp_lac_s5 = emp_s5['LAC']




#%% Calculate value added impacts
value_added_footprint_s5 = np.diag(value_added) @ L_scenario5
value_added_footprint_s5_lac = value_added_footprint_s5 @ Y_scenario5.loc[:, 'LAC']

va_diff_abs_s5 = value_added_footprint_s5_lac.sum() - value_added_footprint_baseline_lac.sum()
va_diff_rel_s5 = va_diff_abs_s5 / value_added_footprint_baseline_lac.sum()


value_added_sum_s5 = value_added_footprint_s5_lac.sum()

delta_VA_s5 = world.query(
    matrices="V",
    scenarios="Scenario 5",
    base_scenario="baseline",
    type='absolute'
    ).loc[(VALUE_ADDED_INDICATOR)]

delta_VA_s5_rel = world.query(
    matrices="V",
    scenarios="Scenario 5",
    base_scenario="baseline",
    type='relative'
    ).loc[(VALUE_ADDED_INDICATOR)]






