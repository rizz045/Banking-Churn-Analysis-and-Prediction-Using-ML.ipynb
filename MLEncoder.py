from sklearn.preprocessing import LabelEncoder
import joblib

class MultiLabelEncoder:
    def __init__(self):
        self.encoders = {}

    def fit(self, df):
        for col in df.columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.encoders[col] = le
            print(f"Classes for {col} class is: ", le.classes_)

    def transform(self, col_names, values):
        if len(col_names) != len(values):
            raise ValueError("Number of columns names must match number of values")
        encoded_values = []

        for col, value in zip(col_names, values):
            if col in self.encoders:
                encoded_values.append(self.encoders[col].transform([value])[0])
            else:
                raise ValueError(f"No encoder found for column: {col}")
        return encoded_values

    def save(self, filename):
        joblib.dump(self, f"{filename}.pkl")  # Save the instance, not the class itself

    @staticmethod
    def load(filename):
        return joblib.load(f"{filename}.pkl")  # Load the saved instance
