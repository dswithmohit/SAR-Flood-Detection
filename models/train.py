from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_model():
    data_dir = Path("data/processed")
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed numpy arrays...")
    X = np.load(data_dir / "X_features.npy")
    y = np.load(data_dir / "y_labels.npy")

    # 1. Split data (using a subset to keep training incredibly fast on your MacBook Air)
    # We take 10% of the 5.5 million pixels, which is still ~550,000 points—plenty for a great model!
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, train_size=0.1, random_state=42, stratify=y
    )

    # 2. Initialize and train the Random Forest
    # n_estimators=50 keeps the tree count low enough to finish in under a minute
    print(f"Training Random Forest on {X_train.shape[0]:,} pixels...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(" Model training complete!")

    # 3. Evaluate the model
    print("\n--- Model Performance Report ---")
    predictions = rf.predict(X_test)
    print(classification_report(y_test, predictions, target_names=["Land", "Water"]))

    # 4. Feature Importance (Which radar signal mattered most?)
    print("--- Feature Importances ---")
    features = ["Pre-Flood (VV)", "Post-Flood (VV)", "Difference"]
    for name, importance in zip(features, rf.feature_importances_):
        print(f"{name}: {importance*100:.2f}%")

    # 5. Save the trained model weights
    model_path = model_dir / "sar_flood_rf_model.pkl"
    joblib.dump(rf, model_path)
    print(f"\nSaved trained model to: {model_path}")

if __name__ == "__main__":
    train_model()