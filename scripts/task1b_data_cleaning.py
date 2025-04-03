import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os
from sklearn.impute import SimpleImputer, KNNImputer

# Create output directory for Task 1b
output_dir = 'task1b_outputs'
os.makedirs(output_dir, exist_ok=True)

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
    'When is your birthday (date)?',
    'How many students do you estimate there are in the room?',
    'Stress_Level_Clean',
    'Sports_Hours_Clean',
    'Give a random number',
    'Time you went to bed Yesterday',
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
    'When is your birthday (date)?': 'Birthday',
    'How many students do you estimate there are in the room?': 'Estimated_Students',
    'Stress_Level_Clean': 'Stress_Level',
    'Sports_Hours_Clean': 'Sports_Hours',
    'Give a random number': 'Random_Number',
    'Time you went to bed Yesterday': 'Bedtime',
    'What makes a good day for you (1)?': 'Good_Day_Factor_1',
    'What makes a good day for you (2)?': 'Good_Day_Factor_2'
})

# Save the cleaned dataset to both the output directory and the data folder
output_csv_path = f'{output_dir}/ODI-2025_cleaned.csv'
data_folder_csv_path = 'data/ODI-2025_cleaned.csv'

df_final.to_csv(output_csv_path, index=False)
df_final.to_csv(data_folder_csv_path, index=False)

print(f"Cleaned dataset saved to {output_csv_path}")
print(f"Cleaned dataset also saved to {data_folder_csv_path}")
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