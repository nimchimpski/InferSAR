#!/usr/bin/env python3
"""
Quick test script to check dataset balance and validation distribution
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path

def check_dataset_balance():
    """Check the balance of training and validation datasets"""
    
    project_dir = Path(__file__).parent.parent
    print(f"Project directory: {project_dir}")
    
    # Check training CSV
    train_csv = project_dir / "data/4final/dataset/flood_train_data.csv"
    val_csv = project_dir / "data/4final/dataset/flood_valid_data.csv"
    
    if not train_csv.exists():
        print(f"Training CSV not found: {train_csv}")
        return
    
    if not val_csv.exists():
        print(f"Validation CSV not found: {val_csv}")
        return
    
    # Load and analyze training data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    print(f"\nTraining dataset:")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Columns: {list(train_df.columns)}")
    
    print(f"\nValidation dataset:")
    print(f"  Total samples: {len(val_df)}")
    print(f"  Columns: {list(val_df.columns)}")
    
    # Check if there are flood label files to analyze distribution
    training_dir = project_dir / "data/4final/training"
    if training_dir.exists():
        print(f"\nTraining directory contents:")
        for item in training_dir.iterdir():
            if item.is_dir():
                count = len(list(item.glob("*")))
                print(f"  {item.name}: {count} files")
    
    # Sample a few files to check label distribution
    print(f"\nSampling first 5 training files:")
    for i, row in train_df.head(5).iterrows():
        print(f"  Row {i}: {row.to_dict()}")

if __name__ == "__main__":
    check_dataset_balance()
