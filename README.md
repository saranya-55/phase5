This project utilizes a Linear Regression model to predict sales based on advertising expenditure on TV and Radio.

Overview
The program reads a dataset named Sales.csv containing information about sales and advertising budgets. It uses the data to train a Linear Regression model and evaluates its performance by predicting sales based on TV and Radio advertising budgets. The Mean Squared Error (MSE) metric is used to assess the model's accuracy.

Requirements
To run this program, ensure you have Python installed along with the following libraries:

pandas
scikit-learn (sklearn)
Setup and Execution
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/sales-prediction.git
Navigate to the project directory:

bash
Copy code
cd sales-prediction
Install the required libraries:

bash
Copy code
pip install pandas scikit-learn
Run the program:

Execute the Python script:

bash
Copy code
python sales_prediction.py
Dataset
The program uses the Sales.csv dataset containing columns for TV, Radio, and Sales. It performs the following data preprocessing steps:

Drops rows with missing values.
Removes duplicate entries.
Model Training
The program splits the dataset into training and testing sets (80/20 split) using the train_test_split function from scikit-learn. It trains a Linear Regression model using the training data with TV and Radio advertising budgets as features (X) and Sales as the target variable (y).

Evaluation
The trained model predicts sales based on the testing dataset's advertising budgets. It computes the Mean Squared Error (MSE) to evaluate the model's predictive performance.

File Structure
sales-prediction.py: The main Python script containing the model training and evaluation code.
data/Sales.csv: Dataset used for training and testing the model.
Note
Ensure the dataset (Sales.csv) is placed in the data directory relative to the script for proper execution.
Feel free to explore, modify, or enhance the code based on your needs.

For any questions or feedback, please contact [Your Name] at [Your Email].
