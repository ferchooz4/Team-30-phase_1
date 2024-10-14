import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer

# Creating the classes

class DataExplorer:
    @staticmethod
    def explore_data(data):
        print("Data Head (Transposed):")
        print(data.head().T)
        print("\nData Description:")
        print(data.describe())
        print("\nData Info:")
        print(data.info())
    
    @staticmethod
    def plot_histograms(data):
        data.hist(bins=15, figsize=(15, 10))
        plt.show()

    @staticmethod
    def plot_correlation_matrix(data):
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.show()

class PredictiveMaintenanceModel:
    def __init__(self, filepath):
        self.filepath = filepath
        self.model_pipeline = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

        # Define the target variable
        self.TARGET = "Machine failure"

        # Define numerical and categorical columns
        self.numerical_columns = ['Air temperature [K]', 'Process temperature [K]', 
                                  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        self.categorical_columns = ['Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

        # Define the preprocessing pipeline
        self.preprocessor = ColumnTransformer([
            ('minmax_scaler', StandardScaler(), self.numerical_columns),
            ('onehot', OneHotEncoder(drop='first'), self.categorical_columns)
        ], remainder='drop')  # Dropping unspecified columns like 'Product ID' and 'UDI'

        # Define the full pipeline
        self.model_pipeline = Pipeline([
            ('preprocessing', self.preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        DataExplorer.explore_data(self.data)
        return self
    
    def preprocess_data(self):
        # Drop columns that are not needed
        self.interim = self.data.drop(['Product ID', 'UDI'], axis=1)
        
        # Define features and target
        X = self.interim.drop(self.TARGET, axis=1)
        y = self.interim[self.TARGET]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return self
    
    def train_model(self):
        self.model_pipeline.fit(self.X_train, self.y_train)
        return self
    
    def evaluate_model(self):
        print("Model Evaluation:")
        y_pred = self.model_pipeline.predict(self.X_test)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(self.y_test))
        disp.plot(cmap='Blues')
        plt.show()
        
        # Classification Report
        report = classification_report(self.y_test, y_pred)
        print("Classification Report:")
        print(report)
        return self
    
    def cross_validate_model(self):
        scores = cross_val_score(self.model_pipeline, self.X_train, self.y_train, cv=5)
        print("Cross-Validation Accuracy Scores:", scores)
        print("Average Accuracy with CV:", np.mean(scores))
        return self

    def plot_feature_importance(self):
        # This method is optional and useful if using models that provide feature importance
        pass

def main():
    filepath = r'D:\Dev\Python Projects\MLOps\phase_1.0\data\raw\ai4i2020.csv'
    model = PredictiveMaintenanceModel(filepath)
    model.load_data()
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.cross_validate_model()

    # Optional: Data Visualization
    # DataExplorer.plot_histograms(model.data)
    # DataExplorer.plot_correlation_matrix(model.data)

if __name__ == '__main__':
    main()
