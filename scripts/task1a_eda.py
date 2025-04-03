import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os

# Create output directory if it doesn't exist
output_dir = 'task1a_outputs'
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

# Load the data
print("Loading dataset...")
df = pd.read_csv('data/ODI-2025.csv', sep=';')

# Basic dataset information
print("\n=== DATASET OVERVIEW ===")
print(f"Number of records: {df.shape[0]}")
print(f"Number of attributes: {df.shape[1]}")
print("\nFirst few rows of the dataset:")
print(df.head())

# Data types and missing values
print("\n=== DATA TYPES AND MISSING VALUES ===")
print(df.dtypes)
print("\nMissing values per column:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})
print(missing_info[missing_info['Missing Values'] > 0])

# Check for empty string entries or special values that might be missing values
print("\nChecking for empty strings or potentially missing values:")
for col in df.columns:
    empty_count = (df[col] == '').sum() + (df[col] == '-').sum() + (df[col] == ' ').sum()
    if empty_count > 0:
        print(f"{col}: {empty_count} empty/dash entries")

# Convert columns to appropriate types for analysis
# This requires some data cleaning first

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

# Applying cleaning functions
print("\n=== CLEANING AND CONVERTING DATA ===")
df['Stress_Level_Clean'] = df['What is your stress level (0-100)?'].apply(clean_stress_level)
df['Sports_Hours_Clean'] = df['How many hours per week do you do sports (in whole hours)? '].apply(clean_sports_hours)

# Program categories
print("\n=== PROGRAM DISTRIBUTION ===")
program_counts = df['What programme are you in?'].value_counts()
print(f"Number of unique programs: {len(program_counts)}")
print("\nTop 10 programs:")
print(program_counts.head(10))

# Course experience
print("\n=== COURSE EXPERIENCE ===")
course_columns = [
    'Have you taken a course on machine learning?',
    'Have you taken a course on information retrieval?',
    'Have you taken a course on statistics?',
    'Have you taken a course on databases?'
]

for col in course_columns:
    print(f"\n{col}:")
    print(df[col].value_counts())

# Gender distribution
print("\n=== GENDER DISTRIBUTION ===")
print(df['What is your gender?'].value_counts())

# ChatGPT usage
print("\n=== CHATGPT USAGE ===")
print(df['I have used ChatGPT to help me with some of my study assignments '].value_counts())

# Numeric attributes analysis
print("\n=== NUMERIC ATTRIBUTES ANALYSIS ===")
print("\nStress level statistics:")
print(df['Stress_Level_Clean'].describe())

print("\nSports hours statistics:")
print(df['Sports_Hours_Clean'].describe())

print("\nEstimated students in room statistics:")
room_students = df['How many students do you estimate there are in the room?'].copy()
# Filter out non-numeric and range values for numeric analysis
room_students = pd.to_numeric(room_students, errors='coerce')
print(room_students.describe())

# Extract and analyze outliers
print("\n=== OUTLIER ANALYSIS ===")

# Stress level outliers (values outside 0-100 range)
stress_outliers = df[~df['Stress_Level_Clean'].between(0, 100)]['Stress_Level_Clean'].dropna()
print(f"\nNumber of stress level outliers (outside 0-100 range): {len(stress_outliers)}")
if len(stress_outliers) > 0:
    print("Sample of stress level outliers:")
    print(stress_outliers.head(5))

# Sports hours outliers - define a reasonable max (e.g., 30 hours per week)
reasonable_max_sports = 30  # Assuming 30 hours per week is a reasonable maximum
sports_outliers = df[df['Sports_Hours_Clean'] > reasonable_max_sports]['Sports_Hours_Clean'].dropna()
print(f"\nNumber of sports hours outliers (> {reasonable_max_sports} hours/week): {len(sports_outliers)}")
if len(sports_outliers) > 0:
    print("Sample of sports hours outliers:")
    print(sports_outliers.head(5))

# Visualizations
print("\n=== CREATING VISUALIZATIONS ===")

# 1. Program distribution (top 10)
plt.figure(figsize=(14, 8))
program_top10 = program_counts.head(10)
ax = sns.barplot(x=program_top10.values, y=program_top10.index, palette=bar_palette)
plt.title('Top 10 Programs', fontsize=16, pad=20)
plt.xlabel('Count', fontsize=14)
plt.ylabel('')  # Hide y-label as it's redundant with the program names
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add count labels to the bars
for i, v in enumerate(program_top10.values):
    ax.text(v + 0.5, i, str(v), color='black', va='center', fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_dir}/program_distribution.png', dpi=300, bbox_inches='tight')

# 2. Gender distribution - with nice colors
plt.figure(figsize=(12, 10))
gender_counts = df['What is your gender?'].value_counts()

# Create the pie chart with improved formatting for small slices
wedges, texts, autotexts = plt.pie(
    gender_counts, 
    labels=[None] * len(gender_counts),  # No labels on the chart itself
    colors=pie_palette,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.85,  # Move the percentage labels closer to the center
    shadow=False,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)

# Make the percentage labels stand out
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

plt.axis('equal')
plt.title('Gender Distribution', fontsize=16, pad=20)

# Create a legend with percentages included
legend_labels = [f"{gender} ({count}, {count/sum(gender_counts)*100:.1f}%)" 
                for gender, count in gender_counts.items()]
plt.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_dir}/gender_distribution.png', dpi=300, bbox_inches='tight')

# 3. Course experience - with improved colors
plt.figure(figsize=(14, 9))
course_data = []
for col in course_columns:
    counts = df[col].value_counts()
    for value, count in counts.items():
        course_data.append({
            'Course': col.replace('Have you taken a course on ', '').replace('?', ''),
            'Response': value,
            'Count': count
        })

course_df = pd.DataFrame(course_data)

# Use a custom palette based on response type
response_palette = {
    'yes': bar_palette[0], 
    'no': bar_palette[1], 
    '1': bar_palette[0], 
    '0': bar_palette[1],
    'mu': bar_palette[2], 
    'sigma': bar_palette[3], 
    'unknown': bar_palette[4],
    'ja': bar_palette[0],
    'nee': bar_palette[1]
}

# Create the plot manually for more control
plt.figure(figsize=(14, 9))
ax = plt.subplot(111)

courses = course_df['Course'].unique()
responses = sorted(course_df['Response'].unique(), key=lambda x: str(x))
width = 0.8 / len(responses)
x = np.arange(len(courses))

for i, response in enumerate(responses):
    response_data = course_df[course_df['Response'] == response]
    response_counts = []
    for course in courses:
        count = response_data[response_data['Course'] == course]['Count'].sum()
        response_counts.append(count)
    
    # Get the color for this response, or use a default
    color = response_palette.get(response, bar_palette[i % len(bar_palette)])
    
    bars = ax.bar(x + i*width - (len(responses)-1)*width/2, response_counts, width, 
                  label=response, color=color, edgecolor='white', linewidth=1)
    
    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(courses, fontsize=12)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Course Experience Distribution', fontsize=16, pad=20)
ax.legend(title='Response', fontsize=12, title_fontsize=14)

plt.tight_layout()
plt.savefig(f'{output_dir}/course_experience.png', dpi=300, bbox_inches='tight')

# 4. Stress level distribution - only 0-100 range with nicer colors
plt.figure(figsize=(12, 8))
# Filter values within the expected range for better visualization
normal_range = df[df['Stress_Level_Clean'].between(0, 100)]['Stress_Level_Clean']
# Create bins from 0 to 100 in steps of 5
bins = np.arange(0, 105, 5)  # 0, 5, 10, ..., 100

# Create custom colors with gradient effect
n_bins = len(bins) - 1
colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, n_bins))

ax = sns.histplot(normal_range, bins=bins, kde=True, color=hist_palette[0], 
                 edgecolor='white', linewidth=1.2, alpha=0.9)

# Add KDE curve with a different color
kde_line = ax.lines[0]
kde_line.set_color(hist_palette[2])
kde_line.set_linewidth(2.5)

plt.title('Stress Level Distribution (0-100 Range)', fontsize=16, pad=20)
plt.xlabel('Stress Level', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/stress_level_distribution.png', dpi=300, bbox_inches='tight')

# 5. Sports hours distribution - only the reasonable range with nicer colors
plt.figure(figsize=(12, 8))
# Filter to a reasonable range for better visualization
reasonable_range = df[df['Sports_Hours_Clean'].between(0, reasonable_max_sports)]['Sports_Hours_Clean']
# Create bins from 0 to reasonable_max_sports in steps of 2
bins = np.arange(0, reasonable_max_sports + 2, 2)  # 0, 2, 4, ..., reasonable_max_sports

ax = sns.histplot(reasonable_range, bins=bins, kde=True, color=hist_palette[1], 
                 edgecolor='white', linewidth=1.2, alpha=0.9)

# Add KDE curve with a different color
kde_line = ax.lines[0]
kde_line.set_color(hist_palette[3])
kde_line.set_linewidth(2.5)

plt.title(f'Weekly Sports Hours Distribution (0-{reasonable_max_sports} Range)', fontsize=16, pad=20)
plt.xlabel('Hours per Week', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/sports_hours_distribution.png', dpi=300, bbox_inches='tight')

# 6. Relationship between stress level and sports hours - only normal ranges with improved styling
plt.figure(figsize=(12, 8))
reasonable_data = df[
    df['Sports_Hours_Clean'].between(0, reasonable_max_sports) & 
    df['Stress_Level_Clean'].between(0, 100)
]

# Create a scatter plot with improved styling
sns.scatterplot(
    x='Sports_Hours_Clean', 
    y='Stress_Level_Clean', 
    data=reasonable_data, 
    color=scatter_palette,
    alpha=0.7,
    s=100,  # Larger point size
    edgecolor='white',
    linewidth=0.5
)

# Add a trend line
sns.regplot(
    x='Sports_Hours_Clean', 
    y='Stress_Level_Clean', 
    data=reasonable_data,
    scatter=False,
    color=hist_palette[3],
    line_kws={'linewidth': 2}
)

plt.title('Stress Level vs Sports Hours (Normal Ranges)', fontsize=16, pad=20)
plt.xlabel('Sports Hours per Week', fontsize=14)
plt.ylabel('Stress Level (0-100)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/stress_vs_sports.png', dpi=300, bbox_inches='tight')

# 7. ChatGPT usage by program (top 10 programs) with improved colors
plt.figure(figsize=(14, 10))
top_programs = program_counts.head(10).index
chatgpt_by_program = pd.crosstab(
    df[df['What programme are you in?'].isin(top_programs)]['What programme are you in?'],
    df[df['What programme are you in?'].isin(top_programs)]['I have used ChatGPT to help me with some of my study assignments ']
)

# Custom colors for ChatGPT usage categories
chatgpt_colors = [bar_palette[0], bar_palette[1], bar_palette[2]]

# Plot the stacked bar chart
ax = chatgpt_by_program.plot(
    kind='bar', 
    stacked=True, 
    color=chatgpt_colors,
    edgecolor='white',
    linewidth=1,
    figsize=(14, 10)
)

# Add data labels on each segment
for i, bar in enumerate(ax.patches):
    width, height = bar.get_width(), bar.get_height()
    if height > 0:  # Only add label if the segment has height
        x, y = bar.get_xy()
        ax.text(x + width/2, y + height/2, f'{int(height)}', 
                ha='center', va='center', fontsize=10, color='black')

plt.title('ChatGPT Usage by Program (Top 10 Programs)', fontsize=16, pad=20)
plt.xlabel('Program', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='ChatGPT Usage', fontsize=12, title_fontsize=14)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/chatgpt_by_program.png', dpi=300, bbox_inches='tight')

# 8. Stress level by gender - only 0-100 range with improved styling
plt.figure(figsize=(12, 8))
normal_stress = df[df['Stress_Level_Clean'].between(0, 100)]

# Create the boxplot with custom styling
ax = sns.boxplot(
    x='What is your gender?', 
    y='Stress_Level_Clean', 
    data=normal_stress, 
    palette=box_palette,
    width=0.6,
    fliersize=5,
    linewidth=1.5
)

# Add individual data points with jitter for better visualization
sns.stripplot(
    x='What is your gender?', 
    y='Stress_Level_Clean', 
    data=normal_stress,
    color='black',
    alpha=0.4,
    size=4,
    jitter=True
)

# Add mean markers
means = normal_stress.groupby('What is your gender?')['Stress_Level_Clean'].mean()
for i, mean_val in enumerate(means):
    ax.plot(i, mean_val, marker='o', color='white', markersize=8, markeredgecolor='black')
    ax.text(i, mean_val + 3, f'Mean: {mean_val:.1f}', ha='center', color='black', fontsize=10)

plt.title('Stress Level by Gender (0-100 Range)', fontsize=16, pad=20)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Stress Level', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/stress_by_gender.png', dpi=300, bbox_inches='tight')

# 9. Word cloud for what makes a good day with enhanced styling
from wordcloud import WordCloud

plt.figure(figsize=(14, 8))

# Combine both good day columns
good_day_text = ' '.join(df['What makes a good day for you (1)?'].dropna().astype(str) + ' ' + 
                         df['What makes a good day for you (2)?'].dropna().astype(str))

# Generate a more colorful word cloud
wordcloud = WordCloud(
    width=1200, 
    height=600, 
    background_color='white',
    max_words=100,
    colormap='viridis',  # Use a colorful colormap
    contour_color='steelblue',
    contour_width=2,
    collocations=True,  # Include word pairs
    prefer_horizontal=0.9,
    min_font_size=10,
    max_font_size=150,
    random_state=42  # For reproducibility
).generate(good_day_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('What Makes a Good Day?', fontsize=20, pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/good_day_wordcloud.png', dpi=300, bbox_inches='tight')

# 10. Sports hours by gender - only 0-30 range with improved styling
plt.figure(figsize=(12, 8))
normal_sports = df[df['Sports_Hours_Clean'].between(0, reasonable_max_sports)]

# Create the boxplot with custom styling
ax = sns.boxplot(
    x='What is your gender?', 
    y='Sports_Hours_Clean', 
    data=normal_sports, 
    palette=box_palette,
    width=0.6,
    fliersize=5,
    linewidth=1.5
)

# Add individual data points with jitter for better visualization
sns.stripplot(
    x='What is your gender?', 
    y='Sports_Hours_Clean', 
    data=normal_sports,
    color='black',
    alpha=0.4,
    size=4,
    jitter=True
)

# Add mean markers
means = normal_sports.groupby('What is your gender?')['Sports_Hours_Clean'].mean()
for i, mean_val in enumerate(means):
    ax.plot(i, mean_val, marker='o', color='white', markersize=8, markeredgecolor='black')
    ax.text(i, mean_val + 0.5, f'Mean: {mean_val:.1f}', ha='center', color='black', fontsize=10)

plt.title(f'Sports Hours by Gender (0-{reasonable_max_sports} Range)', fontsize=16, pad=20)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Hours per Week', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/sports_by_gender.png', dpi=300, bbox_inches='tight')

# Additional interesting findings
print("\n=== INTERESTING FINDINGS ===")

# Relationship between machine learning experience and program
ml_by_program = pd.crosstab(
    df['What programme are you in?'],
    df['Have you taken a course on machine learning?']
)
print("\nMachine learning experience by program (top 5 programs):")
print(ml_by_program.sort_values(by='yes', ascending=False).head(5))

# Relationship between stress level and gender
print("\nAverage stress level by gender:")
stress_by_gender = df.groupby('What is your gender?')['Stress_Level_Clean'].mean().sort_values(ascending=False)
print(stress_by_gender)

# Relationship between ChatGPT usage and stress level
print("\nAverage stress level by ChatGPT usage:")
stress_by_chatgpt = df.groupby('I have used ChatGPT to help me with some of my study assignments ')['Stress_Level_Clean'].mean()
print(stress_by_chatgpt)

# Save the analysis report as a text file
with open(f'{output_dir}/analysis_report.txt', 'w') as f:
    f.write("=== ODI-2025 DATASET ANALYSIS REPORT ===\n\n")
    f.write(f"Number of records: {df.shape[0]}\n")
    f.write(f"Number of attributes: {df.shape[1]}\n\n")
    
    f.write("=== PROGRAM DISTRIBUTION ===\n")
    f.write(f"Number of unique programs: {len(program_counts)}\n")
    f.write("Top 10 programs:\n")
    for program, count in program_counts.head(10).items():
        f.write(f"{program}: {count}\n")
    
    f.write("\n=== GENDER DISTRIBUTION ===\n")
    for gender, count in gender_counts.items():
        f.write(f"{gender}: {count} ({count/len(df)*100:.1f}%)\n")
    
    f.write("\n=== CHATGPT USAGE ===\n")
    chatgpt_counts = df['I have used ChatGPT to help me with some of my study assignments '].value_counts()
    for response, count in chatgpt_counts.items():
        f.write(f"{response}: {count} ({count/len(df)*100:.1f}%)\n")
    
    f.write("\n=== STRESS LEVEL ANALYSIS ===\n")
    f.write("Statistics for all values:\n")
    f.write(f"{df['Stress_Level_Clean'].describe().to_string()}\n\n")
    
    f.write("Statistics for values in 0-100 range:\n")
    f.write(f"{normal_range.describe().to_string()}\n\n")
    
    f.write(f"Number of stress level outliers: {len(stress_outliers)}\n")
    
    f.write("\n=== SPORTS HOURS ANALYSIS ===\n")
    f.write("Statistics for all values:\n")
    f.write(f"{df['Sports_Hours_Clean'].describe().to_string()}\n\n")
    
    f.write(f"Statistics for values in 0-{reasonable_max_sports} range:\n")
    f.write(f"{reasonable_range.describe().to_string()}\n\n")
    
    f.write(f"Number of sports hours outliers: {len(sports_outliers)}\n")

print(f"\nAnalysis complete! All results saved to '{output_dir}/' directory.") 