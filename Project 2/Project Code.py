import pandas as pd
import numpy as np

df = pd.read_csv('Clean Dataset.csv')

print("Dataset Overview:")
print(df.info())
print("\nMissing Values Count:")
print(df.isnull().sum())

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna('Unknown')

def cap_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataframe[column] = np.clip(dataframe[column], lower_bound, upper_bound)
    return dataframe

for col in num_cols:
    df = cap_outliers_iqr(df, col)

summary_stats = df.describe().T[['count', 'mean', '50%']]
summary_stats.rename(columns={'50%': 'median'}, inplace=True)

print("Statistics: ")
print(summary_stats)

total_records = len(df)

if(len(num_cols) > 0):
    primary_metric = num_cols[0]
    total_val = df[primary_metric].sum()
    avg_val = df[primary_metric].mean()
    print("\nBusiness Impact Summary:")
    print(f"Total Observations Analyzed: {total_records:,}")
    print(f"Total {primary_metric}: {total_val:,.2f}")
    print(f"Average {primary_metric}: {avg_val:,.2f}")

date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
if(date_cols):
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])    
    monthly_trend = df.groupby(df[date_col].dt.to_period('M'))[num_cols[0]].sum()
    print("\nMonthly Trend:")
    print(monthly_trend)
