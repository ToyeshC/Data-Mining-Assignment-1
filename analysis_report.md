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