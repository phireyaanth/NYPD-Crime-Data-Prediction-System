
import pandas as pd
import numpy as np

# Step 1: Load Data
data = pd.read_csv('NYPD_Complaint_Data_Historic.csv', low_memory=False)

# Step 2: Drop Unneeded Columns with High Missing Values
data.drop(columns=['HOUSING_PSA', 'TRANSIT_DISTRICT', 'PATROL_BORO', 'KY_CD', 'RPT_DT', 'PD_DESC', 'PD_CD'], inplace=True)

# Step 3: Handle Missing Values
data.fillna({
    'SUSP_RACE': 'UNKNOWN',
    'SUSP_SEX': 'U',
    'VIC_RACE': 'UNKNOWN',
    'VIC_SEX': 'U',
    'BORO_NM': 'UNKNOWN',
    'LOC_OF_OCCUR_DESC': 'UNKNOWN',
    'VIC_AGE_GROUP': 'UNKNOWN',
    'SUSP_AGE_GROUP': 'UNKNOWN'
}, inplace=True)

# Binary Conversion of PARKS_NM, HADEVELOPT, and STATION_NAME
data['PARKS_NM'] = data['PARKS_NM'].notna().astype(int)
data['HADEVELOPT'] = data['HADEVELOPT'].notna().astype(int)
data['STATION_NAME'] = data['STATION_NAME'].notna().astype(int)

# Step 4: Remove Rows with Missing Coordinates
data = data.dropna(subset=['Latitude', 'Longitude'])

# Step 5: Convert Date and Time Columns
data['CMPLNT_FR_DT'] = pd.to_datetime(data['CMPLNT_FR_DT'], errors='coerce')
data['CMPLNT_FR_TM'] = pd.to_datetime(data['CMPLNT_FR_TM'], format='%H:%M:%S', errors='coerce')

# Drop Rows with Missing Date or Time
data = data.dropna(subset=['CMPLNT_FR_DT', 'CMPLNT_FR_TM'])

# Create New Time-Based Features
data['year'] = data['CMPLNT_FR_DT'].dt.year
data['month'] = data['CMPLNT_FR_DT'].dt.month
data['day'] = data['CMPLNT_FR_DT'].dt.day
data['hour'] = data['CMPLNT_FR_TM'].dt.hour
data['weekday'] = data['CMPLNT_FR_DT'].dt.day_name()

# Drop Original Date/Time Columns
data.drop(columns=['CMPLNT_FR_DT', 'CMPLNT_FR_TM'], inplace=True)

# Step 6: Filter Data for Years 2020-2023
data = data[(data['year'] >= 2020) & (data['year'] <= 2023)]

# Step 9: Clean Victim and Suspect Data
valid_age = ['UNKNOWN', '25-44', '<18', '45-64', '65+', '18-24']
data.loc[~data['SUSP_AGE_GROUP'].isin(valid_age), 'SUSP_AGE_GROUP'] = 'UNKNOWN'
data.loc[~data['VIC_AGE_GROUP'].isin(valid_age), 'VIC_AGE_GROUP'] = 'UNKNOWN'

# Step 10: Save Cleaned Data
data.to_csv('cleaned_data_2020_2023.csv', index=False)

print(f"Cleaned data saved with {len(data)} records from 2020 to 2023.")
