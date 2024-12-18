import streamlit as st
import pandas as pd

import Scripts.NeuralNetworkClassifier as nnc
import Scripts.des_pred as des

# Configure the 'Lead Linkedin Url' column as a LinkColumn
column_config = {
    "Lead Linkedin Url": st.column_config.LinkColumn(
        label="LinkedIn Profile",
        display_text="View Profile"
    )
}


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

        # Add in classifications of company descriptions
        df = des.des_classifier(df)

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
                        # st.write("Prediction button clicked!")
                        # st.write(predictions_df)
                        st.data_editor(predictions_df, column_config=column_config, hide_index=True)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
