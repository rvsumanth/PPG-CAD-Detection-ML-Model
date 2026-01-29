import pandas as pd
from .model_definition import HarmonizedCADModel
from .evaluation import evaluate_and_report

def train_model(csv_path):
    print("\nModel training started")
    print("Loading training dataset...")
    df = pd.read_csv(csv_path)

    
    print("Preparing features and labels...")
    X = df.drop(['CAD_LABEL', 'SUBJECT_ID'], axis=1)
    y = df['CAD_LABEL']

    print("Initializing model...")
    model = HarmonizedCADModel()
    model.fit(X, y)


    
    print("Saving trained model...")
    model.save('models/CAD_Harmonized_91_77_Model.pkl')

    # Evaluation after training
    print("Evaluating model performance...")
    evaluate_and_report(
        model=model,
        X=X,
        y=y,
        feature_names=model.feature_names_
    )



    print("Model training, evaluation, and saving completed.")
    return model
