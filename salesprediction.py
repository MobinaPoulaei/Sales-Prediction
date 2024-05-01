import pickle  # Import pickle module for serializing Python objects
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation
from sklearn.model_selection import train_test_split  # Import function to split data into training and test sets
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor from scikit-learn

# Load training data from CSV file into a DataFrame
train = pd.read_csv('train.csv', low_memory=False)
# Load store data from CSV file into a DataFrame
store = pd.read_csv('store.csv')
# Merge the training data with store data on common columns
dataset = pd.merge(train, store, how='inner')

# Convert the 'Date' column to datetime type and extract day, month, and year into separate columns
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset["Day"] = dataset["Date"].dt.day
dataset["Month"] = dataset["Date"].dt.month
dataset["Year"] = dataset["Date"].dt.year
# Remove the 'Date' column after extracting day, month, and year
dataset.drop("Date", axis=1, inplace=True)

# Map values of 'StateHoliday' to more descriptive names
dataset["StateHoliday"] = dataset["StateHoliday"].map({"a": "Public Holiday", "b": "Easter Holiday", "c": "Christmas", "0": "No Holiday"})
# Map 'StoreType' to descriptive names
dataset["StoreType"] = dataset["StoreType"].map({"a": "Type A", "b": "Type B", "c": "Type C", "d": "Type D"})
# Map 'PromoInterval' to custom names and handle missing values
dataset["PromoInterval"] = dataset["PromoInterval"].map({np.nan: "NOTHING", "Jan,Apr,Jul,Oct": "Jan_to_Oct", "Feb,May,Aug,Nov": "Feb_to_Nov",
                                                         "Mar,Jun,Sept,Dec": "Mar_to_Dec"})
# Convert categorical variables into dummy/indicator variables
dataset = pd.get_dummies(dataset)
# Replace any remaining missing values with 0
dataset = dataset.fillna(0)

# Print the first record in the dataset (likely an error, should be dataset.iloc[0])
print(dataset.iloc[0])

# Prepare features by dropping 'Sales' and 'Customers' columns
X = dataset.drop(['Sales', 'Customers'], axis=1)
# Prepare target variable 'Sales'
y = dataset['Sales']
# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train a Random Forest Regressor with 42 trees, utilizing all processor cores
rf = RandomForestRegressor(n_estimators=42, n_jobs=-1)
rf.fit(X_train, y_train)
# Evaluate the model by checking the score on the test set
rf.score(X_test, y_test)

# Save the trained model to a pickle file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf, file)
