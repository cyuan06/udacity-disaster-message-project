# Disaster Response Pipeline Project

## 1.Project Overview

The goal of the project is to apply data engineering skills to analyze the Disaster Response Messages dataset provided by Figure Eight, and build a web application that can help emergency workers analyze incoming messages and sort them into specific categories to speed up aid and contribute to more efficient distribution of people and other resources.

## 2. ETL Pipeline

This is data wrangling pipeline

- Load Messages and Categories dataset

- Merge and Clean dataset

- save cleaned dataset into SQLITE database

## 3. ML Pipeline

This is machine learning pipeline

- Load Dataset from SQL database
- Split dataset into training and testing
- Initialize machine learning model with best parameters
- Train, test and evaluate model
- Print out evaluation results and save model into pickle file

## 4. Flask App

A web application built for user interaction and data visualization
# File Info:

D:\WORKSPACE

|   README.md

|   

+---app

|   |   run.py              //Flask file to run the web application

|   |   

|   \---templates           //contains html file for the web application

|           go.html

|           master.html

|

+---data

|       DisasterResponse.db      // output of the ETL pipeline

|       disaster_categories.csv  // datafile of all the categories

|       disaster_messages.csv    // datafile of all the messages

|       process_data.py          //ETL pipeline scripts

|

\---models

        train_classifier.py      //machine learning pipeline scripts to train and export a classifier
        
- ETL Pipeline Preparation-zh.ipynb # raw code to process dataset
- ML Pipeline Preparation-zh.ipynb # raw code for machine learning model
- README.md

# Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
