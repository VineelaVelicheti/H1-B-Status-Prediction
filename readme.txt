*********Predicting Case Status of H1-B Petitions**********


Steps for implementation:

Unzip the submission folder

**************Data Exploration***********

Load the file "Data_Exploration.py" in Anaconda - Juptyer Notebook
Change the file paths for the importing data files( H-1B_FY2015.xlsx,H-1B_FY2016.xlsx, H-1B_FY2017.xlsx, H-1B_FY2018.xlsx) to the one of unzipped folder
Change the file path at the end of the code for exporting the file to csv in the unzipped folder
Run the Data Exploration file to obtain the master dataset for further processing

********Preprocessing & Classification**********

Load the "Appr1_Preprocessing.py" in Anaconda - Jupyter Notebook
Change the file path for importing the master dataset (the csv obtained from Data Exploration)
Run the file stepwise till the end to obtain the algorithm with best accuracy and factors affecting the prediction of application status


*********Running the sample test file*********
 
To run the sample test file, 

Load the "Appr1_Preprocessing.py" in Anaconda - Jupyter Notebook and import the csv file "H-1B_Test_sample.csv".



********Folders Description **************

3 sub folders along with the Report and Presentation Slides are present. The folders contents are mentioned below in more detail.

1. Data Files : contains two subfolders raw data and combined data. Raw data is the data downloaded from the source website. Combined data includes the csv and excel files exported from "Data_Exploration.py"

2. Experimental Results: Contains various implementation files along with a folder containing screenshots of results.
    Appr2_Preprocessing.py and  Appr3_Preprocessing.py  files contain the two other approaches implemented.
    Other files include implementation of various algorithm combinations.

3. Source Code: This folder contains the main source code files: "Data_Exploration.py" , "Appr1_Preprocessing.py" along with one random test sample to check the evaluation.
  

**********End-of-file*************


