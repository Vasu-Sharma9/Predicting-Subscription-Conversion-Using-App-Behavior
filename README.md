# Predicting-Subscription-Conversion-Using-App-Behavior

# Problem Statement

Many companies in today’s marketplace have mobile apps that offer both free and paid versions. While users often start with the free version, companies want them to convert into paying subscribers (examples: _YouTube Premium, Spotify Premium, Pandora Premium_, etc.).

The challenge is that not every free user is likely to enroll. If the company spends marketing budget equally on all free users, they end up wasting money targeting people who would have subscribed anyway, or those who would never subscribe.

This project is about identifying which free users are _unlikely to convert into paid members_, so that marketing campaigns can be targeted towards the right audience, saving cost and improving conversion rates.

# Project Objective

I was tasked with analyzing user app behavior data collected during the _first 24 hours of app usage_.

The goal is to:

- Perform **Exploratory Data Analysis (EDA)** to understand user patterns.
- Engineer features from app usage (like types of screens visited, time difference between actions, etc.).
- Use this processed dataset to later build _classification models_ that can predict whether a user will enroll into the paid product.

By doing this, the fintech company can launch _targeted campaigns right after the free trial ends_ (since the free trial lasts 24 hours), increasing the chances of successful conversion.

# Dataset Description

The dataset is synthetic but realistic — distributions and patterns mimic real-world app usage in fintech.

It includes user behavior such as:

- Screens visited in the mobile app
- Timestamps for first open and enrollment
- Various app activity features

The dataset is _not clean by default_ and requires preprocessing before applying machine learning.

# What We Did (So Far)

# _Exploratory Data Analysis (EDA)_

- Observed dataset head and statistical summaries
- Cleaned timestamp data and extracted hour of app usage
- Plotted histograms for numerical features
- Studied correlation between independent variables and target
- Visualized correlation heatmap with Seaborn

# _Feature Engineering (Date/Time)_

- Converted string columns into datetime format
- Calculated time differences between `first_open` and `enrolled_date`
- Assumed users taking more than 48 hours to enroll are unlikely customers

# _Feature Engineering (App Screens)_

- Extracted top screens from dataset and created binary variables
- Counted leftover/other screens into a single feature

# _Feature Engineering (Funnels)_

Grouped screens into meaningful categories:

- **SavingCount** → Savings-related screens
- **CMCount** → Credit management screens
- **CCCount** → Credit card screens
- **LoansCount** → Loan-related screens

# _Exported Processed Data_

- Final engineered dataset saved as **`new_appdata10.csv`** for future model training

# Tools and Libraries

- Python (data processing and ML experiments)
- Pandas, Numpy (data handling and manipulation)
- Matplotlib, Seaborn (data visualization)
- Dateutil (date formatting and parsing)

# Repository Structure

├── Dataset/
│ ├── appdata10.csv # Original raw dataset
│ ├── top_screens.csv # List of top app screens
├── new_appdata10.csv # Final cleaned/engineered dataset
├── preprocessing.py # Python script for EDA + feature engineering
├── README.md # Project explanation (this file)
