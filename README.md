# Credit Card Classification using K-Nearest Neighbors (KNN)

This project uses the K-Nearest Neighbors (KNN) algorithm to classify credit card data. The goal is to train a classification model, evaluate its accuracy, and compare different values of \(k\) to see how the number of neighbors affects the results.

## What is KNN?

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression.

For classification, the algorithm looks at the \(k\) closest observations to a new data point and assigns it to the class that appears most frequently among those neighbors.

For example, with \(k=5\), the model looks at the five nearest observations and predicts the majority class.

## How It Works

The code follows these steps:

1. **Load the data:** Read the credit card dataset from a tab-separated text file using pandas.
2. **Inspect the data:** Check the number of rows and columns and preview the first few observations.
3. **Separate predictors and response:** Use all columns except the last one as predictors (`X`) and the last column as the response (`y`).
4. **Standardize the predictors:** Use `StandardScaler` to put the predictor variables on a comparable scale.
5. **Train the model:** Initialize a KNN classifier with `k=5` and fit it to the data.
6. **Make predictions:** Use the trained model to classify observations and check the predicted class counts.
7. **Evaluate accuracy:** Compare the predictions with the actual response values.
8. **Compare different values of k:** Test multiple values of \(k\) using leave-one-out cross-validation and store the accuracy results in a DataFrame.

## Choosing the Value of k

The value of \(k\) determines how many neighboring observations the model considers when making a prediction.

- **Small k:** The model considers fewer neighbors, so predictions can be more sensitive to individual observations.
- **Large k:** The model considers more neighbors, which can produce smoother decision boundaries but may overlook smaller patterns in the data.

The code tests the following values:

`[1, 5, 7, 10, 12, 15, 20, 22, 25]`

The results are stored in `different_k_accuracies_df`, allowing the accuracy of each model to be compared.

## Leave-One-Out Cross-Validation

Instead of evaluating each value of \(k\) on the same observations used for training, the code uses leave-one-out cross-validation.

For each observation, the process is:

1. Leave one observation out of the training data.
2. Train the KNN model using all the remaining observations.
3. Predict the class of the observation that was left out.
4. Repeat until every observation has been tested.
5. Compare all predictions with the actual responses to calculate accuracy.

This gives a more useful estimate of how well different values of \(k\) perform on unseen observations than evaluating the model on its own training data.

## Requirements

- Python 3
- pandas
- scikit-learn

Install the required libraries:

```bash
pip install pandas scikit-learn
```

The dataset file `credit_card_data-headers.txt` must be available in the working directory.

## Running the Code

Save the code in a Python file or run it in a Jupyter Notebook. Make sure the dataset is in the correct location before running it.

## What I Learned

This project helped me practice building a classification model with scikit-learn, standardizing predictor variables, making predictions, and evaluating classification accuracy.

It also helped me understand how the number of neighbors affects KNN and how leave-one-out cross-validation can be used to compare different model settings.

## Note

The initial accuracy calculation uses the same data for training and prediction, so it measures performance on the training data rather than on unseen observations.

The comparison between different values of \(k\) uses leave-one-out cross-validation. However, the scaling is performed before cross-validation, meaning each held-out observation also influences the scaling parameters. For a more rigorous evaluation, the scaler should be fitted separately on each training fold.
