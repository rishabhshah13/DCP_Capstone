from Scripts.Basic_Heuristic_Filtering import heuristic_sort
from Scripts.des_pred import des_classifier
from Scripts.NeuralNetworkClassifier import load_and_infer
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

    # Classify company descriptions
    #df = des_classifier(df)
    df = pd.read_csv("Data/final_df.csv")

    # Perform inference using neural network classifier
    preds_df = load_and_infer(df)

    print(preds_df)
