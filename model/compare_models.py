import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse
import sys
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple models from a JSON config file.")
    parser.add_argument("--config", type=str, default="model/compare_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--out", type=str, default="model/model_comparisons", help="Output directory for the plots.")
    args = parser.parse_args()
    
    comparison_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out = os.path.join(args.out, f"comparison_{comparison_id}")
    return args


def normalize_model_specs(config_data):
    """Convert the JSON config into a list of model specifications.

    Supported formats:
    - New format: a list of objects with keys: name, path, finetuned
    - Legacy format: a mapping of display name -> path
    """
    if isinstance(config_data, list):
        model_specs = []
        for entry in config_data:
            if not isinstance(entry, dict):
                raise ValueError("Each model entry must be an object.")
            if "name" not in entry or "path" not in entry:
                raise ValueError("Each model entry must include 'name' and 'path'.")
            model_specs.append(
                {
                    "name": entry["name"],
                    "path": entry["path"],
                    "finetuned": bool(entry.get("finetuned", False)),
                }
            )
        return model_specs

    if isinstance(config_data, dict):
        return [
            {"name": model_name, "path": run_dir, "finetuned": False}
            for model_name, run_dir in config_data.items()
        ]

    raise ValueError("Configuration file must contain either a list of model specs or a mapping of model names to paths.")


def get_metrics_folder(run_dir, finetuned):
    folder_name = "finetuned" if finetuned else "pretrained"
    preferred_dir = os.path.join(run_dir, folder_name)
    if os.path.isdir(preferred_dir):
        return preferred_dir
    return run_dir

def generate_comparative_plots(model_specs, output_dir="model/model_comparisons", time_point="End Point (t=1)", file_suffix="end"):
    print(f"\nGenerating comparative plots for {time_point}...")
    os.makedirs(output_dir, exist_ok=True)

    df_list = []
    
    # Dynamically load all models specified in the configuration
    for model_spec in model_specs:
        model_name = model_spec["name"]
        run_dir = model_spec["path"]
        finetuned = model_spec["finetuned"]
        metrics_dir = get_metrics_folder(run_dir, finetuned)
        csv_path = os.path.join(metrics_dir, f"continuous_error_violins_{file_suffix}.csv")
        
        if not os.path.exists(csv_path):
            print(f"  [WARNING] Could not find {csv_path}. Skipping '{model_name}'.")
            continue
            
        folder_label = "finetuned" if finetuned else "pretrained"
        print(f"  Loaded data for: {model_name} ({folder_label})")
        df = pd.read_csv(csv_path)
        df['Model'] = model_name
        df_list.append(df)

    if not df_list:
        print("No valid data found to plot. Exiting.")
        return

    # Combine all data
    df_all = pd.concat(df_list, ignore_index=True)

    # Filter to just 3D Euclidean error for the macroscopic comparisons
    df_3d = df_all[df_all['Metric'] == 'Euclidean (3D)'].copy()

    # Clean object names (remove the "(n=X)" so the X-axis labels aren't too crowded)
    df_3d['Object_Clean'] = df_3d['Object'].apply(lambda x: x.split(' (n=')[0])

    # Calculate sorting order: Sort by overall mean error across all models (Descending)
    sort_means = df_3d.groupby('Object_Clean')['Error (cm)'].mean().sort_values(ascending=False)
    order = sort_means.index.tolist()

    # Styling
    sns.set_theme(style="whitegrid")
    
    # Dynamic Palette Generation based on the number of models
    num_models = len(df_list)
    if num_models <= 10:
        palette = sns.color_palette("tab10", num_models) # Good distinct colors for <= 10 models
    else:
        palette = sns.color_palette("husl", num_models)  # Smooth gradient for > 10 models

    # ==========================================
    # Aggregated Violin Plots (All Objects Combined)
    # ==========================================
    plt.figure(figsize=(max(8, num_models * 2), 6)) # Dynamically widen plot if many models
    sns.violinplot(
        data=df_3d,
        x='Model',
        y='Error (cm)',
        hue='Model',
        palette=palette,
        legend=False,
        inner='box',
        cut=0,
        linewidth=1.5
    )
    plt.title(f'Overall Model Performance Distribution\n{time_point}', fontsize=16, fontweight='bold')
    plt.ylabel('Euclidean Error (cm)', fontsize=14, fontweight='bold')
    plt.xlabel('')
    if num_models > 4:
        plt.xticks(rotation=15, ha='right') # Rotate x-labels if there are many models
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'1_overall_violin_comparison_{file_suffix}.png'), dpi=300)
    plt.close()

    # ==========================================
    # Grouped Box Plot (17 Objects x N Models)
    # ==========================================
    plt.figure(figsize=(20, 8))
    sns.boxplot(
        data=df_3d,
        x='Object_Clean',
        y='Error (cm)',
        hue='Model',
        order=order,
        palette=palette,
        showfliers=True,
        linewidth=1.2
    )
    plt.title(f'Object-by-Object Error Distribution (Sorted)\n{time_point}', fontsize=18, fontweight='bold')
    plt.ylabel('Euclidean Error (cm)', fontsize=14, fontweight='bold')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.legend(title='Model architecture', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'2_grouped_boxplot_comparison_{file_suffix}.png'), dpi=300)
    plt.close()

    # ==========================================
    # Grouped Bar Plot (Mean Errors Only)
    # ==========================================
    plt.figure(figsize=(20, 8))
    sns.barplot(
        data=df_3d,
        x='Object_Clean',
        y='Error (cm)',
        hue='Model',
        order=order,
        palette=palette,
        errorbar=None, 
        edgecolor='black'
    )
    plt.title(f'Object-by-Object Mean Error (Sorted)\n{time_point}', fontsize=18, fontweight='bold')
    plt.ylabel('Mean Euclidean Error (cm)', fontsize=14, fontweight='bold')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.legend(title='Model architecture', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'3_grouped_barplot_comparison_{file_suffix}.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        print("Please create a JSON file mapping model display names to their run directories.")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config_data = json.load(f)

    model_specs = normalize_model_specs(config_data)

    print(f"Loaded configuration with {len(model_specs)} models.")

    # Generate plots for Start Point (t=0)
    generate_comparative_plots(
        model_specs=model_specs, 
        output_dir=args.out, 
        time_point="Start Point (t=0)", 
        file_suffix="start"
    )

    # Generate plots for End Point (t=1)
    generate_comparative_plots(
        model_specs=model_specs, 
        output_dir=args.out, 
        time_point="End Point (t=1)", 
        file_suffix="end"
    )
    
    print(f"\nAll comparative plots saved successfully to '{args.out}'.")
