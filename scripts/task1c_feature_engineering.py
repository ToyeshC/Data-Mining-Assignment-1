import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import re

# Create output directory for Task 1c only
output_dir = 'task1c_outputs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs('data', exist_ok=True)  # Keep data directory creation for consistency

# Set the plotting style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Define color palettes for visualizations
bar_palette = sns.color_palette('pastel')
pie_palette = sns.color_palette('pastel')
hist_palette = ["#8dd3c7", "#bebada", "#fb8072", "#80b1d3", "#fdb462"]

print("=== TASK 1C: FEATURE ENGINEERING ===")

# Load the cleaned dataset from Task 1b
print("Loading cleaned dataset...")
df = pd.read_csv('data/ODI-2025_cleaned.csv')
print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} attributes")

# Create a copy for feature engineering
df_engineered = df.copy()

# ==========================================
# 1. TIMESTAMP FEATURES
# ==========================================
print("\n=== STEP 1: EXTRACTING TIMESTAMP FEATURES ===")

# Convert Timestamp to datetime
df_engineered['Timestamp'] = pd.to_datetime(df_engineered['Timestamp'])

# Extract Hour
df_engineered['Hour'] = df_engineered['Timestamp'].dt.hour

print(f"Hours distribution: \n{df_engineered['Hour'].value_counts().sort_index()}")

# ==========================================
# 2. BEDTIME FEATURES
# ==========================================
print("\n=== STEP 2: CREATING BEDTIME FEATURES ===")

# Function to categorize bedtimes
def categorize_bedtime(bedtime):
    if pd.isna(bedtime) or bedtime == '' or not isinstance(bedtime, str):
        return 'Unknown'
    
    try:
        hours, minutes = map(int, bedtime.split(':'))
        time_in_minutes = hours * 60 + minutes
        
        # New categorization:
        # Before 22:00 (1320 mins) = Early Sleeper
        # 22:00-00:00 (1320-1440 mins) = Normal Sleeper
        # After 00:00 (0-360 mins) = Late Sleeper
        if 0 <= time_in_minutes < 360:  # 00:00 to 05:59 - Late sleeper (after midnight)
            return 'Late Sleeper'
        elif 360 <= time_in_minutes < 1320:  # 06:00 to 21:59 - Early sleeper
            return 'Early Sleeper'
        elif 1320 <= time_in_minutes < 1440:  # 22:00 to 23:59 - Normal sleeper
            return 'Normal Sleeper'
        else:
            return 'Unknown'
    except:
        return 'Unknown'

# Apply bedtime categorization
df_engineered['Bedtime_Category'] = df_engineered['Bedtime'].apply(categorize_bedtime)

print(f"Bedtime categories distribution: \n{df_engineered['Bedtime_Category'].value_counts()}")

# ==========================================
# 3. EXPERIENCE FEATURES
# ==========================================
print("\n=== STEP 3: TRANSFORMING EXPERIENCE FEATURES ===")

# Map yes/no/unknown to numeric
exp_mapping = {'yes': 1, 'no': 0, 'unknown': -1}

# Apply mapping to experience columns
experience_cols = ['Machine_Learning_Experience', 'Information_Retrieval_Experience', 
                  'Statistics_Experience', 'Databases_Experience']

for col in experience_cols:
    df_engineered[f'{col}_Numeric'] = df_engineered[col].map(exp_mapping)

# Create Total_Experience feature
df_engineered['Total_Experience'] = df_engineered[[f'{col}_Numeric' for col in experience_cols]].sum(axis=1)

print(f"Total Experience distribution: \n{df_engineered['Total_Experience'].value_counts().sort_index()}")

# ==========================================
# 4. AI ENTHUSIAST FEATURE (UPDATED CRITERIA)
# ==========================================
print("\n=== STEP 4: CREATING AI ENTHUSIAST FEATURE ===")

# Create binary AI_Enthusiast feature - now requiring all 4 courses to be "yes"
df_engineered['AI_Enthusiast'] = (
    (df_engineered['Machine_Learning_Experience'] == 'yes') & 
    (df_engineered['Information_Retrieval_Experience'] == 'yes') &
    (df_engineered['Statistics_Experience'] == 'yes') &
    (df_engineered['Databases_Experience'] == 'yes')
).astype(int)

print(f"AI Enthusiast distribution (must have all 4 courses): \n{df_engineered['AI_Enthusiast'].value_counts()}")

# ==========================================
# 5. AGE CALCULATION
# ==========================================
print("\n=== STEP 5: CALCULATING AGE FEATURES ===")

# Function to extract age from birthday
def extract_age(birthday):
    if pd.isna(birthday) or birthday == '' or birthday == '-':
        return np.nan
    
    try:
        # Handle case where birthday is just month and day (no year)
        if isinstance(birthday, str) and re.match(r'^\d{2}-\d{2}$', birthday):
            return np.nan  # We don't want to make up a year
        
        # Extract year from various date formats
        if isinstance(birthday, str):
            # Try common date formats
            for fmt in ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%B %d, %Y']:
                try:
                    date = datetime.strptime(birthday, fmt)
                    # Calculate age based on survey date (Jan 4, 2025)
                    survey_date = datetime(2025, 1, 4)
                    age = survey_date.year - date.year
                    # Adjust age if birthday hasn't occurred yet in the survey year
                    if (survey_date.month, survey_date.day) < (date.month, date.day):
                        age -= 1
                    return age
                except:
                    continue
                    
            # Try to extract just the year
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', birthday)
            if year_match:
                year = int(year_match.group(1))
                age = 2025 - year
                return age
                
            return np.nan
        else:
            return np.nan
    except:
        return np.nan

# Apply age extraction
df_engineered['Age'] = df_engineered['Birthday'].apply(extract_age)

# Create age categories
def categorize_age(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 20:
        return '<20'
    elif age <= 25:
        return '20-25'
    else:
        return '25+'

df_engineered['Age_Category'] = df_engineered['Age'].apply(categorize_age)

print(f"Age category distribution: \n{df_engineered['Age_Category'].value_counts()}")

# ==========================================
# 6. PHYSICAL ACTIVITY LEVEL
# ==========================================
print("\n=== STEP 6: CATEGORIZING PHYSICAL ACTIVITY ===")

# Function to categorize sports hours
def categorize_sports(hours):
    if pd.isna(hours):
        return 'Unknown'
    elif hours <= 2:
        return 'Low'
    elif hours <= 5:
        return 'Medium'
    else:
        return 'High'

# Apply categorization
df_engineered['Physical_Activity_Level'] = df_engineered['Sports_Hours'].apply(categorize_sports)

print(f"Physical Activity Level distribution: \n{df_engineered['Physical_Activity_Level'].value_counts()}")

# ==========================================
# 7. HAPPINESS CATEGORY FEATURE
# ==========================================
print("\n=== STEP 7: CREATING HAPPINESS CATEGORY FEATURE ===")

# Define categories and their keywords
happiness_categories = {
    'Social': ['friend', 'family', 'love', 'relationship', 'people', 'company', 'partner', 'boyfriend', 'girlfriend', 
               'social', 'connection', 'meet', 'together', 'talk', 'conversation'],
    
    'Personal Growth': ['learn', 'success', 'work', 'achievement', 'accomplish', 'goal', 'study', 'knowledge', 
                       'improve', 'grow', 'progress', 'degree', 'education', 'grade', 'career', 'job'],
    
    'External Factors': ['weather', 'travel', 'food', 'nature', 'sun', 'sunshine', 'sunny', 'warm', 'rain', 'snow', 
                        'temperature', 'climate', 'trip', 'journey', 'meal', 'eat', 'drink', 'coffee', 'environment'],
    
    'Leisure & Activities': ['sport', 'music', 'movie', 'gaming', 'game', 'play', 'hobby', 'activity', 'exercise', 
                           'workout', 'gym', 'run', 'swim', 'relax', 'leisure', 'free time', 'weekend', 'entertainment',
                           'book', 'read', 'watch', 'listen', 'concert', 'festival', 'party', 'vacation', 'holiday', 'sleep']
}

# Function to categorize a happiness factor
def categorize_happiness(factor):
    if pd.isna(factor) or factor == '-':
        return 'Unknown'
    
    factor = str(factor).lower()
    
    for category, keywords in happiness_categories.items():
        if any(keyword in factor for keyword in keywords):
            return category
    
    return 'Other'

# Apply categorization to both factors
df_engineered['Good_Day_Factor_1_Category'] = df_engineered['Good_Day_Factor_1'].apply(categorize_happiness)
df_engineered['Good_Day_Factor_2_Category'] = df_engineered['Good_Day_Factor_2'].apply(categorize_happiness)

# Create the overall Happiness_Category based on both factors
def combine_categories(row):
    cat1 = row['Good_Day_Factor_1_Category']
    cat2 = row['Good_Day_Factor_2_Category']
    
    if cat1 == 'Unknown' and cat2 == 'Unknown':
        return 'Unknown'
    elif cat1 == 'Unknown':
        return cat2
    elif cat2 == 'Unknown':
        return cat1
    elif cat1 == cat2:
        return cat1
    else:
        return 'Mixed'

df_engineered['Happiness_Category'] = df_engineered.apply(combine_categories, axis=1)

print(f"Happiness Category distribution: \n{df_engineered['Happiness_Category'].value_counts()}")
print(f"Factor 1 categories: \n{df_engineered['Good_Day_Factor_1_Category'].value_counts()}")
print(f"Factor 2 categories: \n{df_engineered['Good_Day_Factor_2_Category'].value_counts()}")

# ==========================================
# 8. SAVE ENGINEERED DATASET
# ==========================================
print("\n=== STEP 8: SAVING ENGINEERED DATASET ===")

# Define columns to keep
engineered_columns = [
    # Original columns
    'Timestamp', 'Program', 'Gender', 'ChatGPT_Usage', 
    'Birthday', 'Estimated_Students', 'Stress_Level', 
    'Sports_Hours', 'Random_Number', 'Bedtime',
    'Good_Day_Factor_1', 'Good_Day_Factor_2',
    'Machine_Learning_Experience', 'Information_Retrieval_Experience', 
    'Statistics_Experience', 'Databases_Experience',
    
    # New engineered features
    'Hour', 'Bedtime_Category', 
    'Machine_Learning_Experience_Numeric', 'Information_Retrieval_Experience_Numeric',
    'Statistics_Experience_Numeric', 'Databases_Experience_Numeric',
    'Total_Experience', 'AI_Enthusiast', 'Age', 'Age_Category',
    'Physical_Activity_Level', 'Happiness_Category',
    'Good_Day_Factor_1_Category', 'Good_Day_Factor_2_Category'
]

# Save the final engineered dataset
df_engineered[engineered_columns].to_csv(f'data/ODI-2025_engineered.csv', index=False)
print(f"Engineered dataset saved to data/ODI-2025_engineered.csv")
print(f"Final dataset: {df_engineered.shape[0]} records, {len(engineered_columns)} attributes")

# ==========================================
# 9. CREATE FEATURE SUMMARY
# ==========================================
print("\n=== STEP 9: CREATING FEATURE ENGINEERING SUMMARY ===")

# Visualize the distribution of new categorical features
categorical_features = ['Bedtime_Category', 'Age_Category', 'Physical_Activity_Level', 'Happiness_Category']

plt.figure(figsize=(16, 12))
for i, feature in enumerate(categorical_features):
    plt.subplot(2, 2, i+1)
    counts = df_engineered[feature].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values, palette=bar_palette)
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # Add count labels
    for p, count in zip(ax.patches, counts):
        ax.annotate(f'{count}', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='bottom', 
                    fontsize=10, 
                    color='black',
                    xytext=(0, 5), 
                    textcoords='offset points')
    
plt.tight_layout()
plt.savefig(f'{output_dir}/categorical_features_distribution.png', dpi=300, bbox_inches='tight')

# Visualize the distribution of Total_Experience
plt.figure(figsize=(12, 6))
total_exp_counts = df_engineered['Total_Experience'].value_counts().sort_index()
ax = sns.barplot(x=total_exp_counts.index.astype(str), y=total_exp_counts.values, palette=bar_palette)
plt.title('Distribution of Total Experience')
plt.xlabel('Total Experience Score')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Add count labels
for p, count in zip(ax.patches, total_exp_counts):
    ax.annotate(f'{count}', 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', 
                fontsize=10, 
                color='black',
                xytext=(0, 5), 
                textcoords='offset points')

plt.tight_layout()
plt.savefig(f'{output_dir}/total_experience_distribution.png', dpi=300, bbox_inches='tight')

# Visualize binary features (AI Enthusiast)
plt.figure(figsize=(8, 6))
ai_counts = df_engineered['AI_Enthusiast'].value_counts()
plt.pie(ai_counts.values, labels=['No (0)', 'Yes (1)'], autopct='%1.1f%%', colors=[pie_palette[0], pie_palette[1]])
plt.title(f'Distribution of AI Enthusiast\n(Has all 4 courses)')
plt.annotate(f"Yes: {ai_counts.get(1, 0)} students", xy=(0, -1.2), ha='center')
plt.annotate(f"No: {ai_counts.get(0, 0)} students", xy=(0, -1.4), ha='center')
    
plt.tight_layout()
plt.savefig(f'{output_dir}/ai_enthusiast_distribution.png', dpi=300, bbox_inches='tight')

# Create a feature summary text file
with open(f'{output_dir}/feature_engineering_summary.txt', 'w') as f:
    f.write("=== ODI-2025 DATASET FEATURE ENGINEERING SUMMARY ===\n\n")
    
    f.write("1. TIMESTAMP FEATURES\n")
    f.write("--------------------------------\n")
    f.write("- Extracted Hour (numeric): Hour of the day from timestamp\n")
    f.write(f"- Most common hour: {df_engineered['Hour'].mode()[0]}\n\n")
    
    f.write("2. BEDTIME FEATURES\n")
    f.write("--------------------------------\n")
    f.write("- Created Bedtime_Category (categorical):\n")
    f.write("  - Early Sleeper: before 22:00\n")
    f.write("  - Normal Sleeper: 22:00-00:00\n")
    f.write("  - Late Sleeper: after 00:00\n")
    f.write(f"- Distribution: {dict(df_engineered['Bedtime_Category'].value_counts())}\n\n")
    
    f.write("3. EXPERIENCE FEATURES\n")
    f.write("--------------------------------\n")
    f.write("- Converted categorical experience to numeric (1=yes, 0=no, -1=unknown)\n")
    f.write("- Created Total_Experience: Sum of all experience scores\n")
    f.write(f"- Average Total_Experience: {df_engineered['Total_Experience'].mean():.2f}\n\n")
    
    f.write("4. AI ENTHUSIAST FEATURE\n")
    f.write("--------------------------------\n")
    f.write("- Created AI_Enthusiast binary feature (1 if ALL four course experiences are 'yes')\n")
    f.write(f"- Number of AI enthusiasts: {df_engineered['AI_Enthusiast'].sum()} ({df_engineered['AI_Enthusiast'].mean()*100:.1f}%)\n\n")
    
    f.write("5. AGE FEATURES\n")
    f.write("--------------------------------\n")
    f.write("- Extracted Age from Birthday\n")
    f.write("- Created Age_Category with groups: <20, 20-25, 25+\n")
    f.write(f"- Age distribution: {dict(df_engineered['Age_Category'].value_counts())}\n\n")
    
    f.write("6. PHYSICAL ACTIVITY LEVEL\n")
    f.write("--------------------------------\n")
    f.write("- Categorized Sports_Hours into Physical_Activity_Level:\n")
    f.write("  - Low: 0-2 hours\n")
    f.write("  - Medium: 3-5 hours\n")
    f.write("  - High: 6+ hours\n")
    f.write(f"- Distribution: {dict(df_engineered['Physical_Activity_Level'].value_counts())}\n\n")
    
    f.write("7. HAPPINESS CATEGORY\n")
    f.write("--------------------------------\n")
    f.write("- Categorized Good_Day_Factor_1 and Good_Day_Factor_2 into:\n")
    f.write("  - Social: Friends, Family, Love, Relationships\n")
    f.write("  - Personal Growth: Learning, Success, Work, Achievements\n") 
    f.write("  - External Factors: Weather, Travel, Food, Nature\n")
    f.write("  - Leisure & Activities: Sports, Music, Movies, Gaming\n")
    f.write("  - Mixed: If the two factors belong to different categories\n")
    f.write(f"- Distribution: {dict(df_engineered['Happiness_Category'].value_counts())}\n\n")
    
    f.write("8. DATASET STATISTICS\n")
    f.write("--------------------------------\n")
    f.write(f"Original dataset: {df.shape[0]} records, {df.shape[1]} attributes\n")
    f.write(f"Engineered dataset: {df_engineered.shape[0]} records, {len(engineered_columns)} attributes\n")
    f.write(f"New features created: {len(engineered_columns) - df.shape[1]}\n")

print("\nTask 1c completed! All results saved to the data/ and task1c_outputs/ directories.") 