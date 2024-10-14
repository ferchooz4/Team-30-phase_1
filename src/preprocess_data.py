import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Separate target and features
    TARGET = 'Machine failure'
    
    # Define numerical and categorical columns
    numerical_columns = ['Air temperature [K]', 'Process temperature [K]', 
                         'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    categorical_columns = ['Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    # Define preprocessing: scaling numerical features and encoding categorical features
    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), numerical_columns),
        ('onehot', OneHotEncoder(drop='first'), categorical_columns)
    ], remainder='drop')  # Dropping unspecified columns like 'Product ID' and 'UDI'

    # Features (X) and target (y)
    X = data.drop([TARGET, 'Product ID', 'UDI'], axis=1)
    y = data[TARGET]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply the preprocessing pipeline to the training and test sets
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':
    # Reading file paths from command-line arguments
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    # Save the processed data into separate CSV files
    pd.DataFrame(X_train).to_csv(output_train_features, index=False)
    pd.DataFrame(X_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)
