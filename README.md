# DCP_Capstone
![Alt text](<images/DCP>)
## Overview
Duke Capital Partners (DCP) is initiating a project to develop an internal sourcing tool aimed at identifying Duke University affiliated startup companies seeking funding, which leverages the power of the Duke network to support the university’s entrepreneurial ecosystem.
## Key Feature
Implement advanced filtering system and scoring system which could be directly used upon the raw dataset, enable users (DCP members) to search for Duke-related companies they are interested in (including startups) based on specific criteria such as Duke afiliation, industry, company size, company description etc.
- Filtering system: The goal of this system is to filter out irrelevant features from our raw datasets, and only keep the features that DCP members care about.
- Scoring system: The goal of this system is to score each company on the list based on its features, in order for DCP members to find the companies they are interested in.
## Data Source
- LinkedIn Sales Navigator (used for all model training)
- Techstars Jan 2024 Batch data for description classifier fine-tuning
## Methodology
### Heuristics Model to filter out irrelevant features
- Algorithmic scoring model with 5 weighted criteria based on LinkedIn company factors
- Manually evaluated by DCP team for sorted results aligning with their target companies
- Mostly used for filtering out unwanted metrics from the classification metric
### NLP model quantify company description
- Fine tuning Bert to do a multiclass classification to classify the company description (one of the features of our data) based on part of the collected data with manually assigned labels. (0 = not relevant at all,
	1 = startup companies with Duke connections,
	2 = worth our time!
)
- Preview of training dataset we used:
![Alt text](<images/data>)
### Neural Network classifier
- Final model takes in augmented LinkedIn data and description classifications for softmax ranking of company relevance
## Result
- Company Description Classifier: 0.602 F1-score on validation set
- Neural Network Classifier: 0.84 Recall on positive class predictions
- Streamlit local-hosted UI for simple results pooling


# How to Run

To run this project, follow the steps below:

1. **Create a Python Environment**

   - If you haven't already, create a virtual environment using your preferred method. For example, you can use `virtualenv`:

     ```bash
     virtualenv venv
     ```

   - Activate the virtual environment:

     ```bash
     source venv/bin/activate
     ```

2. **Install Dependencies**

   - Install the required Python packages by running:

     ```bash
     pip install -r requirements.txt
     ```

3. **Open Terminal**
   
4. **Navigate to the root directory of the project**
	![Terminal](images\Steps\step_1.png)

5. **Run the following command:**
   
   ```bash
   streamlit run app.py
   ```

6. **The UI will open automatically in your default web browser. If it doesn't, navigate to [localhost:8501](http://localhost:8501) manually.**

   ![Terminal](images\Steps\step_2.png)

7. **This is how the UI will look like:**
   
   ![UI](images\Steps\step_3.png)
   
   - Button to upload CSV data
   - Button to upload the model
   - Button to predict using the data (will appear after the model is uploaded)

8. **Upload the data using the first button:**
   
   - Select the CSV data file
   - Click on 'Open'
   - 

   ![UI](images\Steps\step_4.png)

9. **Select the model in the same way**

	![UI](images\Steps\step_5.png)

10. **Wait for a while until the documents are processed. There will be a running indicator on the top left.**

	![UI](images\Steps\step_6.png)

11. **After the processing is done, the 'Predict' Button will appear. Click on it and wait.**

	![UI](images\Steps\step_7.png)

12. **Prediction is complete, and you can see the results in the new table**

	![UI](images\Steps\step_8.png)

13. **Click on the LinkedIn Profile, which will redirect you to the link**

Feel free to reach out if you encounter any issues!
