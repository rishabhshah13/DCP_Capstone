from HeuristicFiltering.Basic_Heuristic_Filtering import heuristic_sort
from CompanyDesClassifier.des_pred import des_classifier
import pandas as pd

if __name__ == "__main__":
    # Load in the df of your choice from SalesNavigator exports
    csv_file = input("Enter the file location of the CSV: ") #Data\\all_labelled_data_for_training.csv

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("File not found. Please check the file location and try again.")
        exit()

    # Call the heuristic_sort function
    df = heuristic_sort(csv_file)

    df = des_classifier(df)

    
