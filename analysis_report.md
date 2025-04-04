# ODI-2025 Dataset Analysis Report

This report presents a comprehensive analysis of the ODI-2025 dataset, including exploratory data analysis (Task 1a) and data cleaning procedures (Task 1b).

## 1. Dataset Overview

The ODI-2025 dataset contains survey responses from students across various academic programs, collected on January 4, 2025.

**Basic Information:**
- **Number of records**: 245 (student responses)
- **Number of attributes**: 16 (survey questions)
- **Collection method**: Survey (timestamps indicate data was collected mostly between 12:17-12:20)

## 2. Data Quality Assessment

### Missing Values
The dataset contains missing values in several forms:
- Explicit NaN values
- Empty strings ('')
- Dash characters ('-')
- Special markers like 'unknown', 'not willing to say', etc.

### Data Format Issues
Several columns contain inconsistent formats:
- Date formats vary widely in the birthday column (e.g., '01-01-1888', '31/01/2002', 'September')
- Numeric values with different separators (',' vs '.')
- Numeric values with text (e.g., '4.5 hours', 'Over 9000', 'π©')
- Ranges instead of single values (e.g., '4-5')
- Extremely large values that are likely invalid (e.g., stress levels of 1000, 9999)

### Outliers
- **Stress Level**: 16 values fall outside the expected 0-100 range
- **Sports Hours**: 4 values exceed 30 hours per week, with one extreme outlier of 2.1 billion hours
- **Room Estimates**: Values range from -9999 to 70000 students

## 3. Task 1a: Exploratory Data Analysis

### Academic Programs
- 118 unique program entries in the original dataset
- AI-related programs dominate (33 students in "AI" program)
- Computer Science is the second most common program (9 students)
- Significant redundancy in program names (e.g., "AI" vs "Artificial Intelligence" vs "Master AI")

### Course Experience
The dataset tracks whether students have taken courses in:
- Machine learning: 193 yes, 50 no, 2 unknown
- Information retrieval: 101 "1" (yes), 120 "0" (no), 24 unknown
- Statistics: 170 "mu", 46 "sigma", 29 unknown
- Databases: 170 "ja", 67 "nee", 8 unknown

### Demographics
- **Gender**: Male students (137) outnumber female students (89), with some students identifying as gender fluid (4), intersex (4), non-binary (1), other (1), or choosing not to disclose (9).
- **ChatGPT usage**: Most students (77.1%) report having used ChatGPT for assignments.

### Student Habits & Preferences
- **Sports activity**: Weekly sports hours have a median of 5 hours per week
- **Stress levels**: Reported on a scale from 0-100, with a median of 43.5
- **Good day factors**: Common responses include sunshine, good food, friends, and free time

### Key Findings from EDA
1. **Program Distribution**: AI and Computer Science programs dominate the dataset
2. **Course Experience by Program**: AI students are more likely to have taken machine learning courses
3. **Stress Levels**: Right-skewed distribution, with most students reporting moderate stress (30-60)
4. **ChatGPT Usage**: Students who reported not using ChatGPT had higher average stress levels
5. **Universal Preferences**: Sun and good food are consistently mentioned as factors for a good day

## 4. Task 1b: Data Cleaning

### Extreme Value Removal
- Applied domain knowledge-based thresholds:
  - Stress level: Limited to 0-100 range
  - Sports hours: Limited to 0-30 hours per week
- Removed 18 records with extreme values
- Final dataset: 227 records (versus 245 in the original)

### Course Experience Standardization
- Standardized all course experience responses to a consistent 'yes'/'no'/'unknown' format:
  - Machine Learning: Original 'yes'/'no'/'unknown' format maintained
  - Information Retrieval: Converted '1'/'0'/'unknown' to 'yes'/'no'/'unknown'
  - Statistics: Converted 'mu'/'sigma'/'unknown' to 'yes'/'no'/'unknown'
  - Databases: Converted 'ja'/'nee'/'unknown' to 'yes'/'no'/'unknown'

### Program Name Normalization
- Original dataset had 112 unique program names (after outlier removal)
- Normalized to 16 distinct programs by grouping similar variations
- Major categories include:
  - Artificial Intelligence (93 students)
  - Computer Science (62 students)
  - Bioinformatics (18 students)
  - Business Analytics (16 students)
  - Econometrics (16 students)

### Missing Value Imputation
Two approaches were implemented and compared:
1. **Simple Imputation**:
   - Numeric values: Replaced with median
   - Categorical values: Replaced with most frequent value

2. **KNN Imputation** (selected for final dataset):
   - Used 5 nearest neighbors to estimate missing values
   - Better preserves the natural distributions
   - Accounts for relationships between variables

### Final Cleaned Dataset
- 227 records with 16 attributes
- No missing values
- Consistent data formats and naming conventions

## 5. Key Insights from Both Tasks

1. **Data Quality Challenges**: The dataset contains significant quality issues that required detailed cleaning procedures, particularly in handling extreme values, inconsistent formats, and redundant program names.

2. **Program Distribution**: After normalization, Artificial Intelligence emerges as the clearly dominant program, representing approximately 41% of all students in the cleaned dataset.

3. **Gender Imbalance**: Male students outnumber female students by approximately 3:2 ratio, which may be reflective of gender disparities in computing-related fields.

4. **ChatGPT Adoption**: The high percentage of students using ChatGPT (77.1%) indicates widespread adoption of AI tools in academic settings.

5. **Stress Patterns**: Moderate stress levels (median 43.5) suggest students experience significant but not overwhelming stress. The removal of outliers provides a more accurate representation of the actual stress distribution.

6. **Physical Activity**: The median of 5 hours of sports per week indicates that most students maintain some level of physical activity, though there is considerable variation.

7. **Universal Well-being Factors**: Sunshine, good food, friends, and free time are consistently mentioned as factors for a good day, transcending program and demographic differences.

## 6. Implications for Future Research

1. **Improved Data Collection**: Future surveys should implement stronger validation to prevent extreme values and inconsistent formats.

2. **Program-Specific Analysis**: The normalized program categories enable more meaningful analysis of program-specific patterns in academic experiences and well-being.

3. **AI in Education**: The high ChatGPT usage suggests an opportunity to explore how AI tools are transforming academic practices and potentially affecting student stress levels.

4. **Well-being Interventions**: Insights about what makes a good day for students could inform university well-being initiatives.

5. **Gender Dynamics**: Further research could explore the relationship between gender, program choice, and academic experiences.

The thorough data cleaning performed in Task 1b provides a solid foundation for more sophisticated analyses in the future, including predictive modeling and pattern discovery.

## 7. Task 1c: Feature Engineering

Building on the cleaned dataset from Task 1b, Task 1c implements creative feature engineering to enrich the dataset for better machine learning performance. This process converts raw attributes into more meaningful variables that can potentially improve the performance of machine learning algorithms.

### 7.1 Timestamp Features
- **Hour**: Extracted from the timestamp to capture potential patterns related to time of day
- **Day_of_Week**: Provides information about which day of the week the response was recorded
- Findings: Most responses were collected on the same day within a short timeframe (primarily between hours 12-13)

### 7.2 Sleep Pattern Categorization
- **Bedtime_Category**: Converted raw bedtime values into three categories:
  - Early Sleeper (before 22:00)
  - Normal Sleeper (22:00-00:00)
  - Late Sleeper (after 00:00)
- The bedtime data was first standardized to 24-hour format in the data cleaning stage
- This transformation simplifies the complex time data into interpretable categories that are more easily used in predictive models
- Most students (69.2%) fall into the Early Sleeper category

### 7.3 Experience Quantification
- Converted course experience responses into numeric values:
  - Yes = 1
  - No = 0
  - Unknown = -1
- **Total_Experience**: Created by summing the numeric experience values across all four course types
- The median total experience score is 2, indicating students typically have taken half of the available courses
- This enables quantitative analysis of a student's overall technical background

### 7.4 AI Enthusiasm Indicator
- **AI_Enthusiast**: A binary feature indicating students who have taken all four technical courses:
  - Machine Learning
  - Information Retrieval
  - Statistics
  - Databases
- Only 21.1% of students qualify as AI enthusiasts under this strict definition
- This feature helps identify students with comprehensive technical education across multiple domains

### 7.5 Age Extraction and Categorization
- **Age**: Extracted from birthday field (which was standardized to dd-mm-yyyy format in data cleaning)
- **Age_Category**: Grouped into meaningful categories:
  - <20 years (0.9%)
  - 20-25 years (54.6%)
  - 25+ years (9.3%)
- A significant portion (35.2%) could not have their age determined due to non-standard birthday formats
- This transformation helps create meaningful age groups for analysis despite the highly variable original data

### 7.6 Physical Activity Levels
- **Physical_Activity_Level**: Categorized sports hours into:
  - Low (0-2 hours/week): 19.4%
  - Medium (3-5 hours/week): 35.7%
  - High (6+ hours/week): 44.9%
- Most students engage in significant physical activity, with nearly half in the high category
- This categorical feature provides more interpretable groupings than the raw hours

### 7.7 Happiness Categories
- **Happiness_Category**: Categorized what makes students' days good based on two factors:
  - Social: Friends, family, love, relationships
  - Personal Growth: Learning, success, work, achievements
  - External Factors: Weather, travel, food, nature
  - Leisure & Activities: Sports, music, movies, gaming
- If both factors belong to the same category, that category is assigned
- If they belong to different categories, assigned as "Mixed"
- The distribution reveals:
  - Mixed: 63.0% (most students have diverse sources of happiness)
  - External Factors: 15.9% (many students value environmental elements like sun, food)
  - Other: 16.3% (factors that don't clearly fit into predefined categories)
  - Leisure & Activities: 2.6%
  - Social: 0.9%
  - Personal Growth: 0.4%
- This categorization provides insight into what types of factors contribute to student well-being

### 7.8 Feature Engineering Benefits

The feature engineering process significantly expanded the dataset's analytical potential by:
1. Converting hard-to-analyze raw data into structured features
2. Creating new composite features that capture complex relationships
3. Standardizing and categorizing variables for better interpretability
4. Reducing dimensionality through meaningful categorization

These engineered features provide a strong foundation for subsequent machine learning tasks, especially for:
- Classification problems identifying student categories based on demographics and behaviors
- Predictive modeling of student performance or preferences
- Cluster analysis to identify similar student profiles
- Association rule mining to discover relationships between student characteristics and behaviors

The engineered dataset now contains 30 attributes, including both the original and newly created features, offering a rich basis for subsequent data mining tasks. 