import streamlit as st
import pandas as pd

import Scripts.NeuralNetworkClassifier as nnc


def main():
    """
    Main function to run the Streamlit web app.
    """
    # Title of the web app
    st.title('Duke Capital Partners')

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
            loaded_model = nnc.load_model(load_model_file)
            if loaded_model is not None:
                st.write("Model loaded successfully!")
                # Predict button
                if uploaded_file is not None:
                    if st.button('Predict'):
                        predictions_df = nnc.load_and_infer(loaded_model,df)
                        st.write("Prediction button clicked!")
                        st.write(predictions_df)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
