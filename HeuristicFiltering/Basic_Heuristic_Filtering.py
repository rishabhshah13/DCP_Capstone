import pandas as pd
from datetime import datetime


def heuristic_sort(csv_file):

    df = pd.read_csv(csv_file)

    fill_factors = ['Lead Years In Company', 'Lead Months In Company', 'Lead Months In Position', 'Lead Years In Position']

    for factor in fill_factors:
        df[factor] = df[factor].fillna(0)

    # Calculate total months in position
    df['Lead Total Months In Company'] = df['Lead Years In Company'] * 12 + df['Lead Months In Company']
    df['Lead Total Months In Position'] = df['Lead Years In Position'] * 12 + df['Lead Months In Position']

    def score_row(row):
        # Factor 1: Check if company name is within unwanted titles
        company_name_exclude = ["LLC", "Society", "Foundation", "Attorney", "Advisor", "Consult", "Ventures", "Inc."]
        
        if pd.isnull(row['Company Name']):  # Check if 'Company Name' is NaN
            factor_1_score = -1
            row['factor_1_score'] = factor_1_score
        else:
            if any(company_name.lower() in row['Company Name'].lower() for company_name in company_name_exclude):
                factor_1_score = -2
            else:
                factor_1_score = 0
            row['factor_1_score'] = factor_1_score


        # Factor 2: Remove companies in unwanted industries
        industries_exclude = ["Business Consulting", "Consulting and Coaching Business", "Non-Profit Organization", "Venture Capital", "Private Equity"]
        if pd.notnull(row['Company Industry']):
            if any(industry.lower() in row['Company Industry'].lower() for industry in industries_exclude):
                factor_2_score = -2
            else:
                factor_2_score = 0
        else:
            factor_2_score = -1
        row['factor_2_score'] = factor_2_score

        
        # Factor 3: Remove llc LinkedIn companies
        if pd.notnull(row['Company Linkedin']):
            factor_3_score = -1 if 'llc' in row['Company Linkedin'].lower() else 0
        else:
            factor_3_score = -1
        row['factor_3_score'] = factor_3_score
            

        # Factor 4: Remove unwanted lead job titles
        titles_exclude = ["Web3 Mentor", "Host", "Tutor", "Self-employed", "Instructor"]
        if pd.notnull(row['Lead Job Title']):
            if any(title.lower() in row['Lead Job Title'].lower() for title in titles_exclude):
                factor_4_score = -1
            else:
                factor_4_score = 0
        else:
            factor_4_score = -1
        row['factor_4_score'] = factor_4_score
        
        return row


    def remove_rows_with_negative_factors(df):
        factors = ['factor_1_score', 'factor_2_score', 'factor_3_score', 'factor_4_score']
        negative_rows = df[(df[factors] < 0).any(axis=1)]
        df = df.drop(negative_rows.index)
        return df

    # Factor 5: Remove companies larger than 500
    df = df[df['Company Size'] != '>500']


    # Apply scoring function to each row
    df = df.apply(score_row, axis=1)
    
    df = remove_rows_with_negative_factors(df)

    factors = [col for col in df.columns if col.startswith('factor') and col.endswith('score')]

    # Drop factor columns
    df.drop(columns=factors+['Lead Total Months In Company', 'Lead Total Months In Position'], inplace=True)
    
    return df


    
if __name__ == "__main__":
    # Prompt user for file location
    csv_file = input("Enter the file location of the CSV: ") #Data\\all_labelled_data_for_training.csv

    # Ensure the file location is valid
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("File not found. Please check the file location and try again.")
        exit()

    # Call the heuristic_sort function
    test = heuristic_sort(csv_file)