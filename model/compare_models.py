import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_comparative_plots(cnmp_csv, temp_csv, tedp_csv, output_dir="model/model_comparisons", time_point="End Point (t=1)"):
    print(f"Generating comparative plots for {time_point}...")
    os.makedirs(output_dir, exist_ok=True)

    # Load Data
    df_cnmp = pd.read_csv(cnmp_csv)
    df_cnmp['Model'] = '1. Dual-CNMP (Baseline)'

    df_temp = pd.read_csv(temp_csv)
    df_temp['Model'] = '2. TEMP (Transformer)'

    df_tedp = pd.read_csv(tedp_csv)
    df_tedp['Model'] = '3. TEDP (Diffusion)'

    # Combine all data
    df_all = pd.concat([df_cnmp, df_temp, df_tedp], ignore_index=True)

    # Filter to just 3D Euclidean error for the macroscopic comparisons
    df_3d = df_all[df_all['Metric'] == 'Euclidean (3D)'].copy()

    # Clean object names (remove the "(n=X)" so the X-axis labels aren't too crowded)
    df_3d['Object_Clean'] = df_3d['Object'].apply(lambda x: x.split(' (n=')[0])

    # Calculate sorting order: Sort by overall mean error across all models (Descending)
    sort_means = df_3d.groupby('Object_Clean')['Error (cm)'].mean().sort_values(ascending=False)
    order = sort_means.index.tolist()

    # Styling
    sns.set_theme(style="whitegrid")
    # Professional colors: Red (Baseline), Blue (TEMP), Green (TEDP)
    palette = ["#d9534f", "#5bc0de", "#5cb85c"] 

    # ==========================================
    # Aggregated Violin Plots (All Objects Combined)
    # ==========================================
    plt.figure(figsize=(10, 6))
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
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'1_overall_violin_comparison_{time_point[-4:-1]}.png'), dpi=300)
    plt.close()

    # ==========================================
    # Grouped Box Plot (17 Objects x 3 Models)
    # ==========================================
    plt.figure(figsize=(20, 8))
    sns.boxplot(
        data=df_3d,
        x='Object_Clean',
        y='Error (cm)',
        hue='Model',
        order=order,
        palette=palette,
        showfliers=True, # Shows the outlier dots
        linewidth=1.2
    )
    plt.title(f'Object-by-Object Error Distribution (Sorted)\n{time_point}', fontsize=18, fontweight='bold')
    plt.ylabel('Euclidean Error (cm)', fontsize=14, fontweight='bold')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.legend(title='Model architecture', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'2_grouped_boxplot_comparison_{time_point[-4:-1]}.png'), dpi=300)
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
        errorbar=None, # Removes the variance lines to just show the clean mean bars
        edgecolor='black'
    )
    plt.title(f'Object-by-Object Mean Error (Sorted)\n{time_point}', fontsize=18, fontweight='bold')
    plt.ylabel('Mean Euclidean Error (cm)', fontsize=14, fontweight='bold')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.legend(title='Model architecture', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'3_grouped_barplot_comparison_{time_point[-4:-1]}.png'), dpi=300)
    plt.close()

    print(f"Comparative plots saved to '{output_dir}'")

if __name__ == '__main__':
    cnmp_csv_path = "model/dual_cnmp_latent_alignment/save/run_20260408_235825/continuous_error_violins_start.csv"
    temp_csv_path = "model/transformer_encoded_movement_primitive/save/run_20260408_204033/continuous_error_violins_start.csv"
    tedp_csv_path = "model/transformer_encoded_diffusion_policy/save/run_20260418_194212/continuous_error_violins_start.csv" 

    generate_comparative_plots(cnmp_csv_path, temp_csv_path, tedp_csv_path, time_point="Start Point (t=0)")

    cnmp_csv_path = "model/dual_cnmp_latent_alignment/save/run_20260408_235825/continuous_error_violins_end.csv"
    temp_csv_path = "model/transformer_encoded_movement_primitive/save/run_20260408_204033/continuous_error_violins_end.csv"
    tedp_csv_path = "model/transformer_encoded_diffusion_policy/save/run_20260418_194212/continuous_error_violins_end.csv" 

    generate_comparative_plots(cnmp_csv_path, temp_csv_path, tedp_csv_path, time_point="End Point (t=1)")
