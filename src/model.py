from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def build_model():
    """Original selected classifier; seed added for reproducibility."""
    return Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore')), ('classifier', RandomForestClassifier(random_state=42))])
