import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

wandb_results_path = "model/transformer_encoded_movement_primitive/wandb_results_temp.csv"
df = pd.read_csv(wandb_results_path)

df.dropna(axis=1, how='all', inplace=True)
df.drop(columns=['State', 'Notes', 'Created', 'Runtime', 'Sweep'], inplace=True)
df.sort_values(by='best_val_inv_mse', inplace=True)
df = df.head(20)
df.to_csv("model/transformer_encoded_movement_primitive/top20_wandb_results.csv", index=False)