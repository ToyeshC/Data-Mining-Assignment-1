import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os
from sklearn.impute import SimpleImputer, KNNImputer
import dateutil.parser

# Create output directory for Task 1b only
output_dir = 'task1b_outputs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs('data', exist_ok=True)  # Keep data directory creation for consistency

# Set the plotting style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Custom color palettes for consistency across visualizations
bar_palette = sns.color_palette('pastel')
pie_palette = sns.color_palette('pastel')
hist_palette = ["#8dd3c7", "#bebada", "#fb8072", "#80b1d3", "#fdb462"]
scatter_palette = "#fb8072"
box_palette = sns.color_palette('pastel')

# Load the original data
print("Loading dataset...")
df = pd.read_csv('data/ODI-2025.csv', sep=';')
print(f"Original dataset: {df.shape[0]} records, {df.shape[1]} attributes")

# Create a copy to work with for cleaning
df_clean = df.copy()

# ==========================================
# STEP 1: CLEAN AND STANDARDIZE COLUMNS
# ==========================================

# Standardize column types and handle basic data issues
print("\n=== STEP 1: BASIC DATA CLEANING ===")

# Clean and convert stress level column
def clean_stress_level(value):
    if pd.isna(value) or value == '' or value == '-':
        return np.nan
    
    # Handle various formats and special cases
    if isinstance(value, str):
        # Remove special characters and convert to lowercase
        value = value.lower()
        if value in ['null', 'nan', 'π©', 'i believe i can fly']:
            return np.nan
        if ',' in value:  # Handle European decimal format
            value = value.replace(',', '.')
        
        # Extract numeric part if possible
        match = re.search(r'(\d+(\.\d+)?)', value)
        if match:
            return float(match.group(1))
        return np.nan
    
    return float(value)

# Clean and convert sports hours column
def clean_sports_hours(value):
    if pd.isna(value) or value == '' or value == '-':
        return np.nan
    
    if isinstance(value, str):
        value = value.lower()
        if value in ['zero', '&&&&']:
            return 0
        
        # Handle ranges (take average)
        if '-' in value:
            parts = value.split('-')
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                pass
        
        # Handle hourly format
        if 'h' in value:
            value = value.replace('h', '')
        
        # Handle European decimal format
        if ',' in value:
            value = value.replace(',', '.')
            
        try:
            return float(value)
        except:
            return np.nan
        
    return float(value)

# First pass cleaning - convert to appropriate types
df_clean['Stress_Level_Clean'] = df_clean['What is your stress level (0-100)?'].apply(clean_stress_level)
df_clean['Sports_Hours_Clean'] = df_clean['How many hours per week do you do sports (in whole hours)? '].apply(clean_sports_hours)

# Clean and standardize bedtime to 24-hour format using a more flexible approach
def standardize_bedtime(bedtime):
    if pd.isna(bedtime) or bedtime == '' or bedtime == '-':
        return bedtime  # Keep as is, don't convert to NaN
    
    try:
        # Convert to string and lowercase
        bedtime_str = str(bedtime).lower().strip()
        
        # Handle special cases first
        if bedtime_str in ['night', 'late', 'evening', 'midnight']:
            return '00:00'  # Default for vague night references
        if bedtime_str in ['early', 'morning']:
            return '07:00'  # Default for vague morning references
        
        # Extract AM/PM indicators
        is_pm = any(pm in bedtime_str for pm in ['pm', 'p.m', 'p.m.', 'evening', 'night'])
        is_am = any(am in bedtime_str for am in ['am', 'a.m', 'a.m.', 'morning'])
        
        # Remove all non-numeric characters except for colon and period
        cleaned = re.sub(r'[^0-9:.]', '', bedtime_str)
        
        # Handle different formats
        if ':' in cleaned:  # HH:MM format
            parts = cleaned.split(':')
            if len(parts) >= 2:
                hour, minute = int(float(parts[0])), int(float(parts[1]))
        elif '.' in cleaned:  # HH.MM format
            parts = cleaned.split('.')
            if len(parts) >= 2:
                hour, minute = int(float(parts[0])), int(float(parts[1]))
        else:  # Just hours
            if cleaned:
                hour = int(float(cleaned))
                minute = 0
            else:
                return bedtime  # Unable to parse, keep original
        
        # Special handling for bedtime context - assume most bedtimes are PM
        # If it's 8-12 without AM/PM indicator, assume it's evening (per requirement)
        if not is_am and not is_pm:
            if 8 <= hour <= 12:  # Specific instruction to assume evening time
                is_pm = True
            elif 1 <= hour <= 7:  # Early morning hours after midnight
                is_am = True
        
        # Now apply AM/PM conversion to 24-hour format
        if is_pm and hour < 12:
            hour += 12
        elif is_am and hour == 12:
            hour = 0
        
        # Some sanity checks
        if hour > 23:
            hour = hour % 24
        if minute > 59:
            minute = minute % 60
            
        return f"{hour:02d}:{minute:02d}"
    except Exception as e:
        # If any error occurs, retain the original value rather than making it NaN
        return bedtime

# Clean and standardize birthday to dd-mm-yyyy format using a more flexible approach
def standardize_birthday(birthday, timestamp=None):
    if pd.isna(birthday) or birthday == '' or birthday == '-':
        return birthday  # Keep as is, don't convert to NaN
    
    try:
        # Convert to string and lowercase for consistent processing
        birthday_str = str(birthday).lower().strip()
        
        # Handle relative dates (yesterday, tomorrow, etc.)
        if timestamp and any(word in birthday_str for word in ['yesterday', 'today', 'tomorrow']):
            timestamp_date = pd.to_datetime(timestamp).date()
            
            if 'yesterday' in birthday_str:
                # Yesterday relative to timestamp
                result_date = timestamp_date - pd.Timedelta(days=1)
                return result_date.strftime('%d-%m-%Y')
            elif 'today' in birthday_str:
                # Today relative to timestamp
                return timestamp_date.strftime('%d-%m-%Y')
            elif 'tomorrow' in birthday_str:
                # Tomorrow relative to timestamp
                result_date = timestamp_date + pd.Timedelta(days=1)
                return result_date.strftime('%d-%m-%Y')
        
        # Special case handling for obviously non-date values - keep these as they are
        if birthday_str in ['not willing to say', 'unknown', 'null', 'nan', 'none', 
                           'nothing', 'secret', '?', 'no', 'yesterday', 'today', 'tomorrow']:
            return birthday
        
        # Special case for years with '00' - these are 2000 (per requirement)
        if re.search(r'\/00$|\-00$|\.00$|^00\/|^00\-|^00\.|^00$', birthday_str):
            # Extract day and month if available
            day_month_match = re.search(r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.]00', birthday_str)
            if day_month_match:
                day, month = day_month_match.groups()
                return f"{int(day):02d}-{int(month):02d}-2000"
            else:
                # Just '00' with no other info - default to 01-01-2000
                return "01-01-2000"
        
        # Try dateutil parser for flexible date parsing
        try:
            parsed_date = dateutil.parser.parse(birthday_str, fuzzy=True)
            
            # Special handling for years like '00' which should be 2000
            if parsed_date.year == 2000 and re.search(r'00', birthday_str):
                return parsed_date.strftime('%d-%m-%Y')
            
            # Check if only month and day were provided (no year)
            year_present = re.search(r'\b(19\d{2}|20\d{2})\b', birthday_str) or re.search(r'\b\d{4}\b', birthday_str)
            
            # If no year was provided in the original string, return the date with just month and day
            if not year_present and not parsed_date.year == datetime.now().year:
                # Return just the month and day without imputing a year
                return parsed_date.strftime('%d-%m')
            
            # If we have a valid year, validate it's within a reasonable range
            if 1900 <= parsed_date.year <= 2020:
                return parsed_date.strftime('%d-%m-%Y')
        except:
            pass  # If it fails, continue with other methods
        
        # Check for year '00' in various formats and convert to 2000
        if re.search(r'(\d{1,2})[-/\.](\d{1,2})[-/\.]00', birthday_str):
            match = re.search(r'(\d{1,2})[-/\.](\d{1,2})[-/\.]00', birthday_str)
            if match:
                # Check if it's DD-MM-00 or MM-DD-00 format
                part1, part2 = match.groups()
                if int(part1) <= 31 and int(part2) <= 12:  # Likely DD-MM format
                    return f"{int(part1):02d}-{int(part2):02d}-2000"
                elif int(part2) <= 31 and int(part1) <= 12:  # Likely MM-DD format
                    return f"{int(part2):02d}-{int(part1):02d}-2000"
            
        # Try to handle common formats that might confuse the parser
        # Check for common European format: DD-MM-YYYY or DD/MM/YYYY
        if re.match(r'^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}$', birthday_str):
            parts = re.split(r'[-/\.]', birthday_str)
            if len(parts) == 3:
                day, month, year = parts
                # Special case for '00' as year - convert to 2000
                if year == '00':
                    year = '2000'
                # Ensure 4-digit year for others
                elif len(year) == 2:
                    year = '19' + year if int(year) > 50 else '20' + year
                try:
                    parsed_date = datetime(int(year), int(month), int(day))
                    return parsed_date.strftime('%d-%m-%Y')
                except:
                    pass
        
        # Check for American format: MM-DD-YYYY or MM/DD/YYYY
        if re.match(r'^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}$', birthday_str):
            parts = re.split(r'[-/\.]', birthday_str)
            if len(parts) == 3 and int(parts[0]) <= 12:  # Could be month
                month, day, year = parts
                # Special case for '00' as year - convert to 2000
                if year == '00':
                    year = '2000'
                # Ensure 4-digit year for others
                elif len(year) == 2:
                    year = '19' + year if int(year) > 50 else '20' + year
                try:
                    parsed_date = datetime(int(year), int(month), int(day))
                    return parsed_date.strftime('%d-%m-%Y')
                except:
                    pass
        
        # Check for YYYYMMDD format
        if re.match(r'^\d{8}$', birthday_str):
            year = birthday_str[:4]
            month = birthday_str[4:6]
            day = birthday_str[6:8]
            # Special case for years starting with '00'
            if year == '0000':
                year = '2000'
            try:
                parsed_date = datetime(int(year), int(month), int(day))
                return parsed_date.strftime('%d-%m-%Y')
            except:
                pass
        
        # Check for DDMMYYYY format (no separators)
        if re.match(r'^\d{8}$', birthday_str):
            day = birthday_str[:2]
            month = birthday_str[2:4]
            year = birthday_str[4:8]
            # Special case for years ending with '0000'
            if year == '0000':
                year = '2000'
            try:
                parsed_date = datetime(int(year), int(month), int(day))
                return parsed_date.strftime('%d-%m-%Y')
            except:
                pass
        
        # Extract year-month-day pattern using regex
        year_month_day = re.search(r'(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})', birthday_str)
        if year_month_day:
            year, month, day = year_month_day.groups()
            try:
                parsed_date = datetime(int(year), int(month), int(day))
                return parsed_date.strftime('%d-%m-%Y')
            except:
                pass
        
        # Extract natural language dates (like "September 5")
        month_names = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 
                     'august', 'september', 'october', 'november', 'december']
        
        for i, month_name in enumerate(month_names, 1):
            if month_name in birthday_str:
                # Try to find day
                day_match = re.search(r'\b(\d{1,2})(st|nd|rd|th)?\b', birthday_str)
                day = day_match.group(1) if day_match else '1'  # Default to 1 if no day
                
                # Try to find year, if not found don't impute
                year_match = re.search(r'\b(19\d{2}|20\d{2})\b', birthday_str)
                if year_match:
                    year = year_match.group(1)
                    # Special case for '00' in year
                    if year == '00' or year == '2000':
                        year = '2000'
                    try:
                        parsed_date = datetime(int(year), i, int(day))
                        return parsed_date.strftime('%d-%m-%Y')
                    except:
                        pass
                else:
                    # If only month and day, return without year
                    try:
                        # Just to validate the date is valid
                        parsed_date = datetime(2000, i, int(day))  # Use a leap year as reference
                        return f"{int(day):02d}-{i:02d}"  # Return in DD-MM format
                    except:
                        pass
        
        # If we just have a year, use January 1st as default date with the year
        year_only = re.search(r'\b(19\d{2}|20\d{2}|00)\b', birthday_str)
        if year_only:
            year = year_only.group(1)
            if year == '00':
                year = '2000'
            return f"01-01-{year}"
        
        # If we just have a month, return the month without imputing year
        for i, month_name in enumerate(month_names, 1):
            if month_name in birthday_str:
                return f"01-{i:02d}"  # Return in DD-MM format with day=1
        
        # If nothing worked, preserve the original value
        return birthday
        
    except Exception as e:
        # If any error occurs, retain the original value
        return birthday

# Apply the bedtime standardization
df_clean['Bedtime_Clean'] = df_clean['Time you went to bed Yesterday'].apply(standardize_bedtime)

# Apply the birthday standardization with timestamp info
df_clean['Birthday_Clean'] = df_clean.apply(
    lambda row: standardize_birthday(row['When is your birthday (date)?'], row['Timestamp']), 
    axis=1
)

# Print some statistics about the standardized fields
print("\nBedtime Standardization Stats:")
print(f"Original null values: {df_clean['Time you went to bed Yesterday'].isna().sum()}")
print(f"Cleaned null values: {df_clean['Bedtime_Clean'].isna().sum()}")
print(f"Unique values before cleaning: {df_clean['Time you went to bed Yesterday'].nunique()}")
print(f"Unique values after cleaning: {df_clean['Bedtime_Clean'].nunique()}")

print("\nBirthday Standardization Stats:")
print(f"Original null values: {df_clean['When is your birthday (date)?'].isna().sum()}")
print(f"Cleaned null values: {df_clean['Birthday_Clean'].isna().sum()}")
print(f"Unique values before cleaning: {df_clean['When is your birthday (date)?'].nunique()}")
print(f"Unique values after cleaning: {df_clean['Birthday_Clean'].nunique()}")

# ==========================================
# STEP 2: REMOVING EXTREME AND INCORRECT VALUES
# ==========================================
print("\n=== STEP 2: REMOVING EXTREME AND INCORRECT VALUES ===")

# APPROACH 1: Use domain knowledge to set reasonable thresholds
# For stress level: Keep only values between 0-100
# For sports hours: Keep values between 0-30 hours per week
# For estimated students: Keep values within reasonable range (e.g., 100-1000)

# Print number of outliers before removal
stress_outliers = df_clean[~df_clean['Stress_Level_Clean'].between(0, 100)]['Stress_Level_Clean'].dropna()
sports_outliers = df_clean[df_clean['Sports_Hours_Clean'] > 30]['Sports_Hours_Clean'].dropna()

print(f"Number of stress level outliers (outside 0-100 range): {len(stress_outliers)}")
print(f"Number of sports hours outliers (> 30 hours/week): {len(sports_outliers)}")

# Create a mask for records to keep
keep_stress = df_clean['Stress_Level_Clean'].between(0, 100) | df_clean['Stress_Level_Clean'].isna()
keep_sports = df_clean['Sports_Hours_Clean'].between(0, 30) | df_clean['Sports_Hours_Clean'].isna()

# Apply the filtering
df_clean_filtered = df_clean[keep_stress & keep_sports].copy()

print(f"Records after removing outliers: {df_clean_filtered.shape[0]} (removed {df.shape[0] - df_clean_filtered.shape[0]} records)")

# ==========================================
# STEP 3: STANDARDIZE COURSE EXPERIENCE RESPONSES
# ==========================================
print("\n=== STEP 3: STANDARDIZING COURSE EXPERIENCE RESPONSES ===")

# Create mapping dictionaries for standardization
ml_mapping = {'yes': 'yes', 'no': 'no', 'unknown': 'unknown'}
ir_mapping = {'1': 'yes', '0': 'no', 'unknown': 'unknown'}
stats_mapping = {'mu': 'yes', 'sigma': 'no', 'unknown': 'unknown'}
db_mapping = {'ja': 'yes', 'nee': 'no', 'unknown': 'unknown'}

# Apply mappings
df_clean_filtered['ML_Experience'] = df_clean_filtered['Have you taken a course on machine learning?'].map(ml_mapping)
df_clean_filtered['IR_Experience'] = df_clean_filtered['Have you taken a course on information retrieval?'].map(ir_mapping)
df_clean_filtered['Stats_Experience'] = df_clean_filtered['Have you taken a course on statistics?'].map(stats_mapping)
df_clean_filtered['DB_Experience'] = df_clean_filtered['Have you taken a course on databases?'].map(db_mapping)

# Check standardized course experience distribution
for col in ['ML_Experience', 'IR_Experience', 'Stats_Experience', 'DB_Experience']:
    print(f"\n{col}:")
    print(df_clean_filtered[col].value_counts(dropna=False))

# ==========================================
# STEP 4: NORMALIZE PROGRAM NAMES
# ==========================================
print("\n=== STEP 4: NORMALIZING PROGRAM NAMES ===")

# Get original program distribution
program_counts = df_clean_filtered['What programme are you in?'].value_counts()
print(f"Original number of unique programs: {len(program_counts)}")

# Function to normalize program names
def normalize_program(program):
    if pd.isna(program):
        return np.nan
    
    # Convert to lowercase and strip extra spaces
    program = program.lower().strip()
    
    # Create a mapping based on the user's specific categorization
    
    # Artificial Intelligence category
    ai_programs = [
        "ai", "ai master", "ai and mathematics", "ai masters", "artificial intelligence",
        "artificial intelligence m.sc.", "artificial intelligence master degree", 
        "artificial intelligences", "m artificial intelligence", "msc ai", 
        "msc artificial intelligence", "msc in ai", "main track ai", "master ai", 
        "master ai (uva)", "master ai uva", "master artificial intelligence", 
        "master of ai", "masters ai", "masters artificial intelligence", 
        "masters in ai", "masters in ai (health track)", "masters in artificial intelligence",
        "master's in ai", "master artificial inteligence", "master artifical intelligence"
    ]
    
    # Big Data Engineering category
    bde_programs = ["big data engineering", "big-data engineering"]
    
    # Bioinformatics category
    bio_programs = [
        "bioinformatics", "bioinformatics and systems biology", 
        "bioinformatics's & systems biology", "bioinformatics and system biology",
        "msc bioinformatics and systems biology", "master bioinformatics and systems biology",
        "masters bioinfomatics and systems biology"
    ]
    
    # Biomedical Science category
    biomed_programs = ["biosb master", "biomedical science"]
    
    # Business Analytics category
    ba_programs = [
        "ba", "business analytics", "business analytics master", "business analytics",
        "master business analytics", "master's business analytics"
    ]
    
    # Computational Science category
    comp_sci_programs = [
        "computational science", "master computational science"
    ]
    
    # Computer Science category
    cs_programs = [
        "cs", "computer science", "computer science (seg)", "computer science (joint degree)",
        "computer science fcc", "computer science msc", "msc computer science",
        "msc computer science seg", "msc. in computer science", "master cs", 
        "master computer science", "masters in computer science", "computer science msc",
        "masters in computer science track: software engineering and green it"
    ]
    
    # Econometrics category
    econ_programs = [
        "econometrics", "econometrics & data science", "econometrics & operations research",
        "econometrics - data science", "econometrics and data science", 
        "econometrics and operations research", "msc econometrics", "master econometrics",
        "master econometrics and operations research", "eor"
    ]
    
    # Finance category
    finance_programs = [
        "duisenberg honours programme in finance and technology", "finance and technology",
        "ms finance (finance & technology)", "msc finance: finance and technology honoursprogramme",
        "msc in finance and technology", "quantitative finance", "fintech"
    ]
    
    # Human Language Technology category
    hlt_programs = ["human language technology", "human language technology (rm)"]
    
    # Other specific categories
    humanities_programs = ["humanities research master"]
    iph_programs = ["msc international public health"]
    npn_programs = ["npn"]
    security_programs = ["security"]
    segreen_programs = ["software engineering and green it", "green it"]
    
    # Apply normalization based on program matches
    if any(p in program for p in ai_programs):
        return "Artificial Intelligence"
    
    elif any(p in program for p in bde_programs):
        return "Big Data Engineering"
    
    elif any(p in program for p in bio_programs):
        return "Bioinformatics"
    
    elif any(p in program for p in biomed_programs):
        return "Biomedical Science"
    
    elif any(p in program for p in ba_programs):
        return "Business Analytics"
    
    elif any(p in program for p in comp_sci_programs):
        return "Computational Science"
    
    elif any(p in program for p in cs_programs):
        return "Computer Science"
    
    elif any(p in program for p in econ_programs):
        return "Econometrics"
    
    elif any(p in program for p in finance_programs):
        return "Finance"
    
    elif any(p in program for p in hlt_programs):
        return "Human Language Technology"
    
    elif any(p in program for p in humanities_programs):
        return "Humanities Research Master"
    
    elif any(p in program for p in iph_programs):
        return "International Public Health"
    
    elif any(p in program for p in npn_programs):
        return "Natural and Physical Neuroscience (NPN)"
    
    elif any(p in program for p in security_programs):
        return "Security"
    
    elif any(p in program for p in segreen_programs):
        return "Software Engineering and Green IT"
    
    # Keep as is if no pattern match
    else:
        return program.title()  # Convert to title case for consistency

# Apply normalization
df_clean_filtered['Program_Normalized'] = df_clean_filtered['What programme are you in?'].apply(normalize_program)

# Check normalized program distribution
program_counts_normalized = df_clean_filtered['Program_Normalized'].value_counts()
print(f"Normalized number of unique programs: {len(program_counts_normalized)}")
print("\nTop 10 normalized programs:")
print(program_counts_normalized.head(10))

# Create a mapping from original to normalized program names
print("\n=== CREATING PROGRAM NAME MAPPING ===")
program_mapping = {}
for program in program_counts.index:
    normalized = normalize_program(program)
    if normalized not in program_mapping:
        program_mapping[normalized] = []
    program_mapping[normalized].append(program)

# Save the mapping to a text file
with open(f'{output_dir}/program_name_mapping.txt', 'w') as f:
    f.write("=== PROGRAM NAME NORMALIZATION MAPPING ===\n\n")
    f.write("This file shows how original program names were mapped to normalized categories.\n\n")
    
    for normalized, originals in sorted(program_mapping.items()):
        f.write(f"Normalized: \"{normalized}\"\n")
        f.write("Original names:\n")
        for original in sorted(originals):
            f.write(f"  - \"{original}\"\n")
        f.write("\n")

print(f"Program name mapping saved to {output_dir}/program_name_mapping.txt")

# ==========================================
# STEP 5: IMPUTING MISSING VALUES - APPROACH 1 (SIMPLE)
# ==========================================
print("\n=== STEP 5: IMPUTING MISSING VALUES - APPROACH 1 (SIMPLE) ===")

# Create a copy for Simple Imputation approach
df_imputed_simple = df_clean_filtered.copy()

# For numeric columns: Use median imputation
numeric_cols = ['Stress_Level_Clean', 'Sports_Hours_Clean']
imputer_median = SimpleImputer(strategy='median')
df_imputed_simple[numeric_cols] = imputer_median.fit_transform(df_imputed_simple[numeric_cols])

# For categorical columns: Use most frequent value, but keep 'unknown' as is
categorical_cols = ['ML_Experience', 'IR_Experience', 'Stats_Experience', 'DB_Experience']

# Count NaN values before imputation
print("Missing values before Simple Imputation:")
for col in categorical_cols:
    print(f"{col}: {df_imputed_simple[col].isna().sum()} missing values")

# Apply imputation only to NaN values, preserving 'unknown' values
for col in categorical_cols:
    # Create a mask of NaN values to impute
    nan_mask = df_imputed_simple[col].isna()
    
    if nan_mask.sum() > 0:  # Only apply imputation if there are NaN values
        # Get most frequent value from non-NaN values
        most_freq = df_imputed_simple[col].dropna().mode()[0]
        
        # Apply imputation only to NaN values
        df_imputed_simple.loc[nan_mask, col] = most_freq

# Check results after Simple Imputation
print("\nMissing values after Simple Imputation:")
for col in numeric_cols + categorical_cols:
    print(f"{col}: {df_imputed_simple[col].isna().sum()} missing values")
    
# Count 'unknown' values
print("\n'Unknown' values after Simple Imputation:")
for col in categorical_cols:
    unknown_count = (df_imputed_simple[col] == 'unknown').sum()
    print(f"{col}: {unknown_count} 'unknown' values")

# ==========================================
# STEP 6: IMPUTING MISSING VALUES - APPROACH 2 (KNN)
# ==========================================
print("\n=== STEP 6: IMPUTING MISSING VALUES - APPROACH 2 (KNN) ===")

# Create a copy for KNN Imputation approach
df_imputed_knn = df_clean_filtered.copy()

# For KNN imputation we need to prepare the data
# Convert categorical values to numeric for KNN, with special handling for 'unknown'
df_imputed_knn['ML_Experience_Num'] = df_imputed_knn['ML_Experience'].map({'yes': 1, 'no': 0, 'unknown': 2})
df_imputed_knn['IR_Experience_Num'] = df_imputed_knn['IR_Experience'].map({'yes': 1, 'no': 0, 'unknown': 2})
df_imputed_knn['Stats_Experience_Num'] = df_imputed_knn['Stats_Experience'].map({'yes': 1, 'no': 0, 'unknown': 2})
df_imputed_knn['DB_Experience_Num'] = df_imputed_knn['DB_Experience'].map({'yes': 1, 'no': 0, 'unknown': 2})

# Combine numeric columns for KNN imputation
knn_cols = ['Stress_Level_Clean', 'Sports_Hours_Clean', 
           'ML_Experience_Num', 'IR_Experience_Num', 
           'Stats_Experience_Num', 'DB_Experience_Num']

# Apply KNN imputation
imputer_knn = KNNImputer(n_neighbors=5)
df_imputed_knn[knn_cols] = imputer_knn.fit_transform(df_imputed_knn[knn_cols])

# Convert numeric experience values back to categorical, preserving 'unknown' values
def map_experience_back(value):
    if value < 0.5:
        return 'no'
    elif value > 1.5:  # For values close to 2 (which was 'unknown')
        return 'unknown'
    else:
        return 'yes'

df_imputed_knn['ML_Experience'] = df_imputed_knn['ML_Experience_Num'].apply(map_experience_back)
df_imputed_knn['IR_Experience'] = df_imputed_knn['IR_Experience_Num'].apply(map_experience_back)
df_imputed_knn['Stats_Experience'] = df_imputed_knn['Stats_Experience_Num'].apply(map_experience_back)
df_imputed_knn['DB_Experience'] = df_imputed_knn['DB_Experience_Num'].apply(map_experience_back)

# Check results after KNN Imputation
print("Missing values after KNN Imputation:")
for col in numeric_cols + categorical_cols:
    print(f"{col}: {df_imputed_knn[col].isna().sum()} missing values")

# Count 'unknown' values
print("\n'Unknown' values after KNN Imputation:")
for col in categorical_cols:
    unknown_count = (df_imputed_knn[col] == 'unknown').sum()
    print(f"{col}: {unknown_count} 'unknown' values")

# Drop temporary numeric columns used for KNN
df_imputed_knn.drop(['ML_Experience_Num', 'IR_Experience_Num', 
                     'Stats_Experience_Num', 'DB_Experience_Num'], axis=1, inplace=True)

# ==========================================
# STEP 7: COMPARE IMPUTATION APPROACHES
# ==========================================
print("\n=== STEP 7: COMPARING IMPUTATION APPROACHES ===")

# Compare stress level distributions
plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
sns.histplot(df_clean_filtered['Stress_Level_Clean'].dropna(), bins=20, kde=True, color=hist_palette[0])
plt.title('Stress Level - Original (Without Outliers)', fontsize=14)
plt.xlim(0, 100)

plt.subplot(1, 3, 2)
sns.histplot(df_imputed_simple['Stress_Level_Clean'], bins=20, kde=True, color=hist_palette[1])
plt.title('Stress Level - Simple Imputation', fontsize=14)
plt.xlim(0, 100)

plt.subplot(1, 3, 3)
sns.histplot(df_imputed_knn['Stress_Level_Clean'], bins=20, kde=True, color=hist_palette[2])
plt.title('Stress Level - KNN Imputation', fontsize=14)
plt.xlim(0, 100)

plt.tight_layout()
plt.savefig(f'{output_dir}/stress_level_imputation_comparison.png', dpi=300, bbox_inches='tight')

# Compare sports hours distributions
plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
sns.histplot(df_clean_filtered['Sports_Hours_Clean'].dropna(), bins=15, kde=True, color=hist_palette[0])
plt.title('Sports Hours - Original (Without Outliers)', fontsize=14)
plt.xlim(0, 30)

plt.subplot(1, 3, 2)
sns.histplot(df_imputed_simple['Sports_Hours_Clean'], bins=15, kde=True, color=hist_palette[1])
plt.title('Sports Hours - Simple Imputation', fontsize=14)
plt.xlim(0, 30)

plt.subplot(1, 3, 3)
sns.histplot(df_imputed_knn['Sports_Hours_Clean'], bins=15, kde=True, color=hist_palette[2])
plt.title('Sports Hours - KNN Imputation', fontsize=14)
plt.xlim(0, 30)

plt.tight_layout()
plt.savefig(f'{output_dir}/sports_hours_imputation_comparison.png', dpi=300, bbox_inches='tight')

# ==========================================
# STEP 8: VISUALIZE NORMALIZED PROGRAM DISTRIBUTION
# ==========================================
print("\n=== STEP 8: VISUALIZING NORMALIZED PROGRAM DISTRIBUTION ===")

# Original vs. Normalized Program Distribution
plt.figure(figsize=(16, 12))

# Original programs (top 15)
plt.subplot(2, 1, 1)
top_orig_programs = df_clean_filtered['What programme are you in?'].value_counts().head(15)
ax1 = sns.barplot(x=top_orig_programs.values, y=top_orig_programs.index, palette=bar_palette)
plt.title('Top 15 Original Program Names', fontsize=16, pad=20)
plt.xlabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add count labels
for i, v in enumerate(top_orig_programs.values):
    ax1.text(v + 0.5, i, str(v), color='black', va='center', fontsize=12)

# Normalized programs (top 15)
plt.subplot(2, 1, 2)
top_norm_programs = df_clean_filtered['Program_Normalized'].value_counts().head(15)
ax2 = sns.barplot(x=top_norm_programs.values, y=top_norm_programs.index, palette=bar_palette)
plt.title('Top 15 Normalized Program Names', fontsize=16, pad=20)
plt.xlabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add count labels
for i, v in enumerate(top_norm_programs.values):
    ax2.text(v + 0.5, i, str(v), color='black', va='center', fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_dir}/program_normalization_comparison.png', dpi=300, bbox_inches='tight')

# ==========================================
# STEP 9: PREPARE THE FINAL CLEANED DATASET
# ==========================================
print("\n=== STEP 9: PREPARING FINAL CLEANED DATASET ===")

# Based on the imputation comparison, select the best approach
# For this task, we'll go with KNN imputation as it preserves the distribution better
# and considers relationships between variables
df_final = df_imputed_knn.copy()

# Prepare final dataset with only the relevant columns
final_columns = [
    'Timestamp',
    'Program_Normalized',
    'ML_Experience',
    'IR_Experience',
    'Stats_Experience',
    'DB_Experience',
    'What is your gender?',
    'I have used ChatGPT to help me with some of my study assignments ',
    'Birthday_Clean',
    'How many students do you estimate there are in the room?',
    'Stress_Level_Clean',
    'Sports_Hours_Clean',
    'Give a random number',
    'Bedtime_Clean',
    'What makes a good day for you (1)?',
    'What makes a good day for you (2)?'
]

# Rename columns for clarity
df_final = df_final[final_columns].rename(columns={
    'Program_Normalized': 'Program',
    'ML_Experience': 'Machine_Learning_Experience',
    'IR_Experience': 'Information_Retrieval_Experience',
    'Stats_Experience': 'Statistics_Experience',
    'DB_Experience': 'Databases_Experience',
    'What is your gender?': 'Gender',
    'I have used ChatGPT to help me with some of my study assignments ': 'ChatGPT_Usage',
    'Birthday_Clean': 'Birthday',
    'How many students do you estimate there are in the room?': 'Estimated_Students',
    'Stress_Level_Clean': 'Stress_Level',
    'Sports_Hours_Clean': 'Sports_Hours',
    'Give a random number': 'Random_Number',
    'Bedtime_Clean': 'Bedtime',
    'What makes a good day for you (1)?': 'Good_Day_Factor_1',
    'What makes a good day for you (2)?': 'Good_Day_Factor_2'
})

# Save the cleaned dataset to the data folder
data_folder_csv_path = 'data/ODI-2025_cleaned.csv'
df_final.to_csv(data_folder_csv_path, index=False)
print(f"Cleaned dataset saved to {data_folder_csv_path}")
print(f"Final dataset: {df_final.shape[0]} records, {df_final.shape[1]} attributes")

# ==========================================
# STEP 10: DATA CLEANING SUMMARY
# ==========================================
print("\n=== STEP 10: DATA CLEANING SUMMARY ===")

# Create a summary text file
with open(f'{output_dir}/data_cleaning_report.txt', 'w') as f:
    f.write("=== ODI-2025 DATASET CLEANING REPORT ===\n\n")
    
    f.write("1. EXTREME VALUE REMOVAL APPROACH\n")
    f.write("--------------------------------\n")
    f.write("- Approach: Domain knowledge-based thresholds\n")
    f.write("- Stress level: Limited to 0-100 range (valid scale range)\n")
    f.write("- Sports hours: Limited to 0-30 hours per week (reasonable maximum)\n")
    f.write(f"- Removed {df.shape[0] - df_clean_filtered.shape[0]} records with extreme values\n")
    f.write("- Justification: These thresholds represent realistic boundaries for the attributes,\n")
    f.write("  and values outside these ranges likely represent data entry errors or\n")
    f.write("  intentional extreme responses that would skew analysis.\n\n")
    
    f.write("2. IMPUTATION APPROACHES COMPARISON\n")
    f.write("--------------------------------\n")
    f.write("a) Simple Imputation:\n")
    f.write("   - Numeric columns: Replaced with median values\n")
    f.write("   - Categorical columns: Replaced with most frequent values\n")
    f.write("   - Pros: Simple, fast, and intuitive\n")
    f.write("   - Cons: Doesn't account for relationships between variables\n\n")
    
    f.write("b) KNN Imputation:\n")
    f.write("   - Used K=5 nearest neighbors to estimate missing values\n")
    f.write("   - Pros: Preserves relationships between variables, maintains distribution shape\n")
    f.write("   - Cons: More computationally intensive, sensitive to feature scaling\n\n")
    
    f.write("   Selected approach: KNN Imputation\n")
    f.write("   Justification: KNN imputation better preserves the natural distributions and\n")
    f.write("   accounts for relationships between variables, which is important for maintaining\n")
    f.write("   the overall structure of the dataset. This results in more realistic imputed values\n")
    f.write("   compared to simple median/mode imputation.\n\n")
    
    f.write("3. ADDITIONAL CLEANING STEPS\n")
    f.write("--------------------------------\n")
    f.write("a) Standardized course experience responses:\n")
    f.write("   - Machine Learning: 'yes'/'no' format\n")
    f.write("   - Information Retrieval: Converted '1'/'0' to 'yes'/'no'\n")
    f.write("   - Statistics: Converted 'mu'/'sigma' to 'yes'/'no'\n")
    f.write("   - Databases: Converted 'ja'/'nee' to 'yes'/'no'\n\n")
    
    f.write("b) Normalized program names:\n")
    f.write(f"   - Original unique programs: {len(program_counts)}\n")
    f.write(f"   - Normalized unique programs: {len(program_counts_normalized)}\n")
    f.write("   - Applied rules to group similar programs (e.g., 'AI', 'Artificial Intelligence', 'Master AI')\n")
    f.write("   - Standardized capitalization and removed extra spaces\n\n")
    
    f.write("4. FINAL DATASET STATISTICS\n")
    f.write("--------------------------------\n")
    f.write(f"Original dataset: {df.shape[0]} records, {df.shape[1]} attributes\n")
    f.write(f"Cleaned dataset: {df_final.shape[0]} records, {df_final.shape[1]} attributes\n")
    f.write("Changes:\n")
    f.write(f"- Removed {df.shape[0] - df_clean_filtered.shape[0]} records with extreme values\n")
    f.write("- Standardized course experience formats\n")
    f.write("- Normalized program names\n")
    f.write("- Imputed missing values using KNN approach\n")
    f.write("- Renamed columns for clarity\n")

print("\nTask 1b completed! All results saved to the 'task1b_outputs/' directory.")

# Add visualization of stress level by ChatGPT usage
plt.figure(figsize=(12, 8))

# Create the boxplot with custom styling
ax = sns.boxplot(
    x='ChatGPT_Usage', 
    y='Stress_Level', 
    data=df_final, 
    palette=box_palette,
    width=0.6,
    fliersize=5,
    linewidth=1.5
)

# Add individual data points with jitter for better visualization
sns.stripplot(
    x='ChatGPT_Usage', 
    y='Stress_Level', 
    data=df_final,
    color='black',
    alpha=0.4,
    size=4,
    jitter=True
)

# Add mean markers with values
means = df_final.groupby('ChatGPT_Usage')['Stress_Level'].mean()
for i, mean_val in enumerate(means):
    ax.plot(i, mean_val, marker='o', color='white', markersize=8, markeredgecolor='black')
    ax.text(i, mean_val + 3, f'Mean: {mean_val:.1f}', ha='center', color='black', fontsize=10)

# Customize labels
plt.title('Stress Level by ChatGPT Usage (After Imputation)', fontsize=16, pad=20)
plt.xlabel('ChatGPT Usage', fontsize=14)
plt.ylabel('Stress Level', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/stress_by_chatgpt_imputed.png', dpi=300, bbox_inches='tight') 