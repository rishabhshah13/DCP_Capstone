import torch.nn as nn
import datetime
import numpy as np
import pickle
import torch
import pandas as pd

# List of columns to be one-hot encoded
hot_encode_columns = [
    'Lead Job Title',
    'Company Size',
    'Company Industry',
    'Company Li Company Type',
    'Company Location Country Name',
    'Email Status'
]

def load_and_infer(df):
    """
    Loads a trained neural network model, preprocesses the input dataframe, performs inference,
    and returns predictions for each company in the dataframe.
    
    Args:
    df (DataFrame): Input dataframe containing company data.
    
    Returns:
    DataFrame: Predictions dataframe containing company names and predicted relevancy scores.
    """
    # Define the neural network model architecture
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            # Define layers
            self.layer1 = nn.Linear(11, 16)
            self.layer2 = nn.Linear(16, 16)
            self.layer3 = nn.Linear(16, 16)
            self.layer4 = nn.Linear(16, 16)
            self.layer5 = nn.Linear(16, 16)
            self.layer6 = nn.Linear(16, 16)
            self.layer7 = nn.Linear(16, 16)
            self.output_layer = nn.Linear(16, 2)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            # Define forward pass
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.relu(self.layer3(x))
            x = self.relu(self.layer4(x))
            x = self.relu(self.layer5(x))
            x = self.relu(self.layer6(x))
            x = self.relu(self.layer7(x))
            x = self.softmax(self.output_layer(x))
            return x

    # Load the trained model
    model = NeuralNetwork()
    model.load_state_dict(torch.load('Model/neural_network_model.pth'))
    model.eval()

    # Preprocess dataframe
    df_encoded = pd.DataFrame()

    # Calculate the current year
    current_year = datetime.datetime.now().year
    
    # Feature engineering: compute years since company founded
    df_encoded['Years Since Company Founded'] = current_year - df['Company Founded In']

    # Feature engineering: compute total months in position and total months in company
    df_encoded['Total Months In Position'] = df['Lead Years In Position'] * 12 + df['Lead Months In Position']
    df_encoded['Total Months In Company'] = df['Lead Years In Company'] * 12 + df['Lead Months In Company']

    # Copy numerical columns and hot encode categorical columns
    for column in hot_encode_columns+['Company Followers', 'Company Des Relevant Score']:
        df_encoded[column] = df[column]

    # Perform one-hot encoding for categorical columns
    for column in hot_encode_columns:
        # Load the encoder
        encoder_filename = f'encoders/{column}_encoder.pkl'  # Assuming the encoders are saved in a folder named 'encoders'
        with open(encoder_filename, 'rb') as file:
            encoder = pickle.load(file)
        
        # Encode the column in df_encoded
        encoded_column = encoder.transform(df_encoded[[column]])
        
        # Replace the column in df_encoded with the encoded values
        df_encoded[column] = encoded_column

    # Fill missing values with 0
    df_encoded.fillna(0, inplace=True)

        # Convert dataframe to tensor
    df_numeric = df_encoded.apply(pd.to_numeric, errors='coerce')
    df_numeric.dropna(inplace=True)
    df_numeric.reset_index(drop=True, inplace=True)  # Reset index after dropping NaN values
    
    df_array = df_numeric.values.astype(np.float32)
    df_tensor = torch.tensor(df_array)

    # Perform inference
    with torch.no_grad():
        output = model(df_tensor)
    
    # Convert probabilities to raw probabilities
    probabilities = output.numpy()

    # Resetting index for df to match df_numeric
    df.reset_index(drop=True, inplace=True)
    
    # Create a dataframe with Company Name and Predictions
    company_names = df['Company Name'].iloc[df_numeric.index]
    predictions_df = pd.DataFrame({'Company Name': company_names, 'Predicted_Relevancy': probabilities[:, 1]})

    # Sort by Predicted_Relevancy
    predictions_df = predictions_df.sort_values(by='Predicted_Relevancy', ascending=False)
    predictions_df.reset_index(drop=True, inplace=True)
    
    return predictions_df


if __name__ == "__main__":
    csv_file_name = 'Data/final_df.csv'  #Data\all_labelled_data_for_training.csv
    df = pd.read_csv(csv_file_name, dtype={'Company Size': str})

    preds_df = load_and_infer(df)
    print(preds_df)
