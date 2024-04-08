import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import datetime
import numpy as np

# Function to load model
def load_model(uploaded_model):
    if uploaded_model is not None:
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super(NeuralNetwork, self).__init__()
                self.layer1 = nn.Linear(966, 16)
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
                x = self.relu(self.layer1(x))
                x = self.relu(self.layer2(x))
                x = self.relu(self.layer3(x))
                x = self.relu(self.layer4(x))
                x = self.relu(self.layer5(x))
                x = self.relu(self.layer6(x))
                x = self.relu(self.layer7(x))
                x = self.softmax(self.output_layer(x))
                return x

        model = NeuralNetwork()
        model.load_state_dict(torch.load(uploaded_model))
        model.eval()
        st.text("Model loaded successfully!")
        print("Model Loaded")
        return model        
    else:
        st.text("No model file uploaded")
        return None


def load_and_infer(model,df):
    # Load saved model
    

    # Preprocess dataframe
    df_encoded = pd.DataFrame()

    current_year = datetime.datetime.now().year
    df_encoded['Years Since Company Founded'] = current_year - df['Company Founded In']

    df_encoded['Total Months In Position'] = df['Lead Years In Position'] * 12 + df['Lead Months In Position']

    df_encoded['Total Months In Company'] = df['Lead Years In Company'] * 12 + df['Lead Months In Company']
    hot_encode_columns = [
        'Lead Job Title',
        'Company Size',
        'Company Industry',
        'Company Li Company Type',
        'Company Location Country Name',
        'Email Status'
    ]

    for column in hot_encode_columns+['Company Followers', 'Company Des Relevant Score']:
        df_encoded[column] = df[column]

    df_encoded = pd.get_dummies(df_encoded, columns=hot_encode_columns)

    df_encoded.fillna(0, inplace=True)

    # Convert dataframe to tensor
    df_numeric = df_encoded.apply(pd.to_numeric, errors='coerce')
    df_numeric.dropna(inplace=True)
    df_array = df_numeric.values.astype(np.float32)
    df_tensor = torch.tensor(df_array)

    # Perform inference
    with torch.no_grad():
        output = model(df_tensor)
    
    # Convert probabilities to raw probabilities
    probabilities = output.numpy()

    # Create a dataframe with Company Name and Predictions
    company_names = df['Company Name'].iloc[df_numeric.index]
    predictions_df = pd.DataFrame({'Company Name': company_names, 'Predicted_Relevancy': probabilities[:, 1]})

    # Sort by Predicted_Relevancy
    predictions_df = predictions_df.sort_values(by='Predicted_Relevancy', ascending=False)

    predictions_df = predictions_df.reset_index(drop=True)
    
    return predictions_df


def main():
    # Title of the web app
    st.title('CSV Viewer')

    # File uploader widget to upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # File uploader widget to upload model
    load_model_file = st.file_uploader("Upload Model", type=["pth"])

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Display the DataFrame
        st.write(df)

    if load_model_file is not None:
        # Load model
        try:
            loaded_model = load_model(load_model_file)
            if loaded_model is not None:
                st.write("Model loaded successfully!")
                # Predict button
                if uploaded_file is not None:
                    if st.button('Predict'):
                        predictions_df = load_and_infer(loaded_model,df)
                        st.write("Prediction button clicked!")
                        st.write(predictions_df)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
