import os

print("Running Data Preprocessing...")
os.system("python -m src.data_preprocessing")

print("Running Feature Engineering...")
os.system("python -m src.feature_engineering")

print("Training Model...")
os.system("python -m src.train_model")

print("Evaluating Model...")
os.system("python -m src.evaluate_model")

print("Pipeline Completed")