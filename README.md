Link Prediction

The project purpose is to analyze and predict a coauthor dataset using the link prediction algorithm. We created an appropiate training dataset, in which 4 kings of algorithms where tested, and select the best model based on the Kaggle competition 

Installing
- networkx
- numpy
- sklearn

Code
1. 1_Pre_process_json.py: pre process the data in the json input file to transform it to a csv file, where the data for each author is a row with its characteristics

2. 2_Pre_get_links.py: creates the training dataset given the network and the defined measures and assumptions explained in the report. Runs an random process to select the edges between authors that will represent the no coauthorship examples to analize

3. 3_test_preprocessing.py: creates the test dataset with the same measures as 2. but with the given edges in the test-public.csv file

4. Runs the three different models with logistic regression and Cross Validation (no transformation/no standarized, standarized, standarized/transformed) for selecting parameters and hyperparameters. Selects best model and generates output file for kaggle test

5. Runs the two different models with neural network and Cross Validation (standarized, standarized/transformed) for selecting parameters and hyperparameters. Selects best model and generates output file for kaggle test

6. Runs the three different models with random forest and Cross Validation (standarized, standarized/transformed) for selecting parameters and hyperparameters. Selects best model and generates output file for kaggle test

6. Runs the three different models with SVM and Cross Validation (standarized, standarized/transformed) for selecting parameters and hyperparameters. Selects best model and generates output file for kaggle test