#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Set a clean, modern style
sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)

# Use a cohesive color palette
TRANSFORMATION_COLORS = {
    'Combined': '#7F3C8D',
    'Split': '#11A579',
    'Scaled': '#3969AC',
    'Spatial': '#F2B701',
    'Normal': '#999999'
}

df = pd.read_csv("benchmark_results.csv")
# Add filter for K Min Vehicles
df = df[df['K Min Vehicles'] <= 100]

if 'Transformation' in df.columns:
    df['Transformation'] = df['Transformation'].str.title()

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return encoded

fig_list = []

# Gap Distribution by Transformation Type
if 'Transformation' in df.columns and 'Gap (%)' in df.columns:
    plt.figure(figsize=(10, 6))
    counts = df['Transformation'].value_counts()
    labels = [f'{t}\n(n={counts[t]})' for t in df['Transformation'].unique()]

    sns.boxplot(x='Transformation', y='Gap (%)', data=df,
                palette=[TRANSFORMATION_COLORS[t] for t in df['Transformation'].unique()],
                saturation=0.8)
    sns.stripplot(x='Transformation', y='Gap (%)', data=df, color='black', alpha=0.7, jitter=True, size=6)
    plt.title('Gap Distribution by Transformation Type', weight='bold', pad=20)
    plt.xlabel('')
    plt.ylabel('Gap (%)', weight='bold')
    plt.xticks(range(len(labels)), labels)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    fig_list.append(('Gap Distribution by Transformation Type', fig_to_base64()))


# Truck Load Percentage by Transformation Type
load_cols = ['Truck Load % (Min)', 'Truck Load % (Max)', 'Truck Load % (Avg)']
if all(col in df.columns for col in load_cols) and 'Transformation' in df.columns:
    plt.figure(figsize=(10, 6))
    counts = df['Transformation'].value_counts()
    labels = [f'{t}\n(n={counts[t]})' for t in df['Transformation'].unique()]

    melt_df = df.melt(id_vars=['Transformation'], value_vars=load_cols, 
                      var_name='Load Metric', value_name='Load %')
    sns.boxplot(x='Transformation', y='Load %', hue='Load Metric', data=melt_df, 
                palette='RdBu', saturation=0.8)
    plt.title('Truck Load Percentage by Transformation Type', weight='bold', pad=20)
    plt.xlabel('')
    plt.ylabel('Load Percentage (%)', weight='bold')
    plt.xticks(range(len(labels)), labels)
    plt.legend(title='Metric', frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    fig_list.append(('Truck Load Percentage by Transformation', fig_to_base64()))


# Cumulative Distribution of Gap
if 'Gap (%)' in df.columns:
    plt.figure(figsize=(10, 6))
    sorted_gaps = sorted(df['Gap (%)'].dropna())
    if len(sorted_gaps) > 0:
        yvals = [(i+1)/len(sorted_gaps) for i in range(len(sorted_gaps))]
        plt.step(sorted_gaps, yvals, where='post', color='#3969AC', linewidth=2)
        plt.title('Cumulative Distribution of Gap (%)', weight='bold', pad=20)
        plt.xlabel('Gap (%)', weight='bold')
        plt.ylabel('Cumulative Proportion of Instances', weight='bold')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        fig_list.append(('Cumulative Distribution of Gap', fig_to_base64()))

# Distribution of Gaps by Transformation
if 'Gap (%)' in df.columns and 'Transformation' in df.columns:
    for transformation in df['Transformation'].unique():
        plt.figure(figsize=(10, 6))
        subset = df[df['Transformation'] == transformation]
        sns.histplot(subset['Gap (%)'], bins=8, 
                     color=TRANSFORMATION_COLORS[transformation], 
                     edgecolor='white', alpha=0.8)
        mean_val = subset["Gap (%)"].mean()
        median_val = subset["Gap (%)"].median()
        plt.axvline(mean_val, color='r', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_val:.1f}%')
        plt.axvline(median_val, color='g', linestyle='--', linewidth=2, 
                    label=f'Median: {median_val:.1f}%')
        plt.title(f'Distribution of Gaps - {transformation}', weight='bold', pad=20)
        plt.xlabel('Gap (%)', weight='bold')
        plt.ylabel('Frequency', weight='bold')
        plt.legend(title=f'Instances: {len(subset)}', frameon=False, loc='upper left')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        fig_list.append((f'Distribution of Gaps - {transformation}', fig_to_base64()))


# Correlation between N Customers and Gap (aggregated)
if 'N Customers' in df.columns and 'Gap (%)' in df.columns:
    for transformation in df['Transformation'].unique():
        plt.figure(figsize=(10, 6))
        sns.regplot(x='N Customers', y='Gap (%)', data=df, 
                    scatter_kws={'alpha':0.7}, line_kws={'color':'red'})
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.title(f'Correlation between N Customers and Gap (%) - Transformation {transformation}', weight='bold', pad=20)
        plt.xlabel('N Customers (%)', weight='bold')
        plt.ylabel('Gap (%) - Log Scale', weight='bold')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        fig_list.append((f'Correlation between N Customers and Gap (%) - Transformation {transformation}', fig_to_base64()))

# Correlation between K Min Vehicles and Gap (aggregated)
if 'K Min Vehicles' in df.columns and 'Gap (%)' in df.columns:
    for transformation in df['Transformation'].unique():
        plt.figure(figsize=(10, 6))
        sns.regplot(x='K Min Vehicles', y='Gap (%)', data=df, 
                    scatter_kws={'alpha':0.7}, line_kws={'color':'red'})
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.title(f'Correlation between K Min Vehicles and Gap (%) - Transformation {transformation}', weight='bold', pad=20)
        plt.xlabel('K Min Vehicles', weight='bold')
        plt.ylabel('Gap (%) - Log Scale', weight='bold')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        fig_list.append((f'Correlation between K Min Vehicles and Gap (%) - Transformation {transformation}', fig_to_base64()))



# Correlation between Vehicle Capacity and Gap (aggregated)
if 'Vehicle Capacity' in df.columns and 'Gap (%)' in df.columns:
    # Convert Vehicle Capacity to numeric, dropping any non-numeric values
    df['Vehicle Capacity'] = pd.to_numeric(df['Vehicle Capacity'], errors='coerce')
    
    for transformation in df['Transformation'].unique():
        subset_df = df[df['Transformation'] == transformation].dropna(subset=['Vehicle Capacity', 'Gap (%)'])
        
        if not subset_df.empty:
            plt.figure(figsize=(10, 6))
            sns.regplot(x='Vehicle Capacity', y='Gap (%)', data=subset_df, 
                        scatter_kws={'alpha':0.7}, line_kws={'color':'red'})
            plt.yscale('log')  # Set y-axis to logarithmic scale
            plt.title(f'Correlation between Vehicle Capacity and Gap (%) - {transformation}', weight='bold', pad=20)
            plt.xlabel('Vehicle Capacity', weight='bold')
            plt.ylabel('Gap (%) - Log Scale', weight='bold')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            fig_list.append((f'Correlation between Vehicle Capacity and Gap (%) - {transformation}', fig_to_base64()))







# Combined plots: Best Solution vs FSM and Lower Bound Gap Distribution
if all(col in df.columns for col in ['K Min Vehicles', 'Total Vehicles', 'Transformation']) and 'Gap (%)' in df.columns:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Left plot: Best Solution vs FSM for all Transformations
    max_val = max(df['Total Vehicles'].max(), df['K Min Vehicles'].max()) + 5
    ax1.plot([0, max_val], [0, max_val], 'r--', label='Equal to Minimum', linewidth=2)
    
    sns.scatterplot(x='K Min Vehicles', y='Total Vehicles', 
                    hue='Transformation', data=df,
                    palette=TRANSFORMATION_COLORS,
                    s=120, alpha=0.8, edgecolor='white', linewidth=1,
                    ax=ax1)
    
    ax1.set_xlabel('Minimum Vehicles (K_min)', weight='bold')
    ax1.set_ylabel('Calculated Vehicles', weight='bold')
    ax1.set_title('Best-Known Solutions vs FSM - All Transformations', weight='bold', pad=20)
    ax1.legend(frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right plot: Lower Bound Gap Distribution
    lower_bound_transformations = ['Scaled', 'Split', 'Spatial']
    subset = df[df['Transformation'].isin(lower_bound_transformations)]
    
    for transformation in lower_bound_transformations:
        trans_data = subset[subset['Transformation'] == transformation]
        sns.histplot(data=trans_data, x='Gap (%)', bins=8,
                     color=TRANSFORMATION_COLORS[transformation],
                     alpha=0.6,
                     label=transformation,
                     ax=ax2)
    
    ax2.set_title('Lower Bound Gap Distribution', weight='bold', pad=20)
    ax2.set_xlabel('Gap (%)', weight='bold')
    ax2.set_ylabel('Frequency', weight='bold')
    ax2.legend(title='', frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig_list.append(('Combined Analysis', fig_to_base64()))


html_content = "<html><head><title>Analysis Results</title></head><body>"
html_content += "<h1>Analysis Results</h1>"

for title, img_data in fig_list:
    html_content += f"<h2>{title}</h2>"
    html_content += f'<img src="data:image/png;base64,{img_data}" alt="{title}" style="max-width:800px; display:block; margin-bottom:30px;">'

html_content += "</body></html>"

with open("results.html", "w") as f:
    f.write(html_content)

print("All figures generated and embedded in results.html. Open results.html to view.")

