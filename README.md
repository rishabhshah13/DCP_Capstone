# DCP_Capstone
![Alt text](<Screenshot 2024-04-04 at 9.43.04 PM.png>)
## Overview
Duke Capital Partners (DCP) is initiating a project to develop an internal sourcing tool aimed at identifying Duke University affiliated startup companies seeking funding, which leverages the power of the Duke network to support the university’s entrepreneurial ecosystem.
## Key Feature
-Implement advanced filtering system and scoring system which could be directly used upon the raw dataset, enable users (DCP members) to search for Duke-related companies they are interested in (including startups) based on specific criteria such as Duke afiliation, industry, company size, company description etc.
- Filtering system: The goal of this system is to filter out irrelevant features from our raw datasets, and only keep the features that DCP members care about.
- Scoring system: The goal of this system is to score each company on the list based on its features, in order for DCP members to find the companies they are interested in.
## Data Source
- LinkedIn Sales Navigator (used for the model training)
- Harmonic AI
## Methodology
### Heuristics Model to filter out irrelevant features
- Algorithmic scoring model with 10 weighted criteria based on LinkedIn company factors
- Manually evaluated by DCP team for sorted results aligning with their target companies
- Mostly used for filtering out unwanted metrics from the classification metric
### NLP model quantify company description
- Fine tuning Bert to do a multiclass classification to classify the company description (one of the features of our data) based on part of the collected data with manually assigned labels. (0 = not relevant at all,
	1 = startup companies with Duke connections,
	2 = worth our time!
)
- Preview of training dataset we used:
![Alt text](<Screenshot 2024-04-04 at 9.54.03 PM.png>)
### Nerual network classifier
- Our final model to put all things together and get a score of each company  based on all filtered features from the dataset.
### Result