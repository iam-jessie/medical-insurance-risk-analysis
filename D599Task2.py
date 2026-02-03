import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# ----------------------------------------------------------------------------------
# Part I: Univariate and Bivariate Statistical Analysis and Visualization
# ----------------------------------------------------------------------------------
plt.style.use('dark_background')
sns.set_context("notebook", font_scale=1.1)

WUWA_PALETTE = ["#4CC9F0", "#F4D35E", "#EF476F", "#A8DADC"]
sns.set_palette(WUWA_PALETTE)

# Configures plot params
plt.rcParams['axes.facecolor'] = '#121212'
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['grid.color'] = '#444444'
plt.rcParams['grid.linestyle'] = ':'

def load_data(filepath):
    """Loads dataset and handles errors."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data. Shape: {df.shape}")
        missing = df.isnull().sum()
        if missing.any():
            print("Missing values detected:")
            print(missing[missing > 0])

            df = df.dropna()
            print(f"Data after dropping missing: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        return None

def plot_univariate_distributions(df, quant_vars, qual_vars):
    """Generates univariate visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution of Key Variables', color='white', fontsize=18, weight='light')

    # BMI Distribution
    sns.histplot(df[quant_vars[0]], kde=True, ax=axes[0, 0], color=WUWA_PALETTE[0], element="step", fill=True, alpha=0.3, lw=1)
    mean_bmi = df[quant_vars[0]].mean()
    axes[0, 0].axvline(mean_bmi, color='white', linestyle='--', linewidth=1)
    axes[0, 0].text(mean_bmi + 1, 5, f'Avg: {mean_bmi:.1f}', color='white', style='italic')
    axes[0, 0].set_title(f'BMI Distribution (Average: {mean_bmi:.1f})', color='white', pad=20)
    axes[0, 0].grid(True, linestyle=':', alpha=0.4)

    # Charges Distribution
    sns.histplot(df[quant_vars[1]], kde=True, ax=axes[0, 1], color=WUWA_PALETTE[1], element="step", fill=True, alpha=0.3, lw=1)
    mean_charges = df[quant_vars[1]].mean()
    axes[0, 1].axvline(mean_charges, color='white', linestyle='--', linewidth=1)
    axes[0, 1].text(mean_charges + 1000, 5, f'Avg: ${mean_charges:,.0f}', color='white', style='italic')
    axes[0, 1].set_title(f'Medical Charges (Skewed Distribution)', color='white', pad=20)
    axes[0, 1].grid(True, linestyle=':', alpha=0.4)

    # Smoker Count
    smoker_counts = df[qual_vars[0]].value_counts()
    bars = axes[1, 0].bar(smoker_counts.index, smoker_counts.values, color=[WUWA_PALETTE[2], WUWA_PALETTE[3]], edgecolor='white', alpha=0.8)
    axes[1, 0].set_title(f'Smoker vs Non-Smoker Count', color='white', pad=20)
    axes[1, 0].grid(axis='y', linestyle=':', alpha=0.4)
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 10,
                        f'{height}', ha='center', va='bottom', color='white', fontsize=11)

    # Region Count
    region_counts = df[qual_vars[1]].value_counts()
    bars2 = axes[1, 1].bar(region_counts.index, region_counts.values, color=WUWA_PALETTE, edgecolor='white', alpha=0.8)
    axes[1, 1].set_title(f'Beneficiaries by Region', color='white', pad=20)
    axes[1, 1].grid(axis='y', linestyle=':', alpha=0.4)
    for bar in bars2:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height}', ha='center', va='bottom', color='white', fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('univariate_distributions.png', dpi=150)
    plt.show()

def plot_bivariate_relationships(df):
    """Generates bivariate visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Relationship Analysis', color='white', fontsize=18, weight='light')

    # BMI vs Charges
    sns.scatterplot(x='bmi', y='charges', data=df, hue='smoker', palette=[WUWA_PALETTE[2], WUWA_PALETTE[3]], ax=axes[0, 0], alpha=0.8, s=70)
    axes[0, 0].set_title('Impact of BMI on Cost (Smoker Highlight)', color='white', pad=15)
    axes[0, 0].legend(title='Smoker Status', facecolor='#121212', edgecolor='white', loc='upper left')
    axes[0, 0].grid(True, linestyle=':', alpha=0.4)

    # Charges by Smoker
    sns.boxplot(x='smoker', y='charges', data=df, ax=axes[0, 1], palette=[WUWA_PALETTE[2], WUWA_PALETTE[3]], boxprops=dict(alpha=0.8))
    axes[0, 1].set_title('Cost Variance: Smoker vs Non-Smoker', color='white', pad=15)
    axes[0, 1].grid(True, linestyle=':', alpha=0.4)

    # BMI by Region
    sns.boxplot(x='region', y='bmi', data=df, ax=axes[1, 0], palette=WUWA_PALETTE, boxprops=dict(alpha=0.8))
    axes[1, 0].set_title('Regional BMI Comparison', color='white', pad=15)
    axes[1, 0].grid(True, linestyle=':', alpha=0.4)

    # Charges by Region
    sns.boxplot(x='region', y='charges', data=df, ax=axes[1, 1], palette=WUWA_PALETTE, boxprops=dict(alpha=0.8))
    axes[1, 1].set_title('Regional Cost Comparison', color='white', pad=15)
    axes[1, 1].grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('bivariate_relationships.png', dpi=150)
    plt.show()

def generate_descriptive_statistics(df):
    """Prints statistics tables."""
    # Quantitative
    print("\n" + "="*60)
    print("Descriptive Statistics (Quantitative)")
    print("="*60)
    print(df.describe().to_string())

    # Qualitative
    print("\n" + "="*60)
    print("Descriptive Statistics (Qualitative)")
    print("="*60)
    
    qual_cols = df.select_dtypes(include=['object']).columns
    for col in qual_cols:
        count = df[col].value_counts()
        percent = df[col].value_counts(normalize=True) * 100
        stats = pd.DataFrame({'Count': count, 'Percentage': percent})
        print(f"\n--- Variable: {col.upper()} ---")
        print(stats.to_string())
    
    #Part II Parametric Test
def perform_parametric_test(df):
    """
    Parametric Test: Two-sample T-test for BMI (Male vs Female).
    Justification: BMI is normally distributed.
    """
    print("\n" + "="*60)
    print("Parametric Test Results (BMI by Sex)")
    print("="*60)

    # Segregate groups
    if 'sex' not in df.columns or 'bmi' not in df.columns:
        print("Error: Required columns for parametric test not found.")
        return
    male_bmi = df[df['sex'] == 'male']['bmi']
    female_bmi = df[df['sex'] == 'female']['bmi']
    if male_bmi.empty or female_bmi.empty:
        print("Error: One or both groups are empty for parametric test.")
        return
    # Assumption Check: Levene's Test
    stat_var, p_var = stats.levene(male_bmi, female_bmi)
    equal_var = p_var > 0.05
    print(f"Assumption Check (Levene's P-value): {p_var:.4f}")
    # Run T-Test
    t_stat, p_value = stats.ttest_ind(male_bmi, female_bmi, equal_var=equal_var)
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_value:.4f}")
    print(f"Male Mean:   {male_bmi.mean():.2f}")
    print(f"Female Mean: {female_bmi.mean():.2f}")
    if p_value < 0.05:
        print("Result: Significant Difference (Reject Null)")
    else:
        print("Result: No Significant Difference (Fail to Reject Null)")
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.title('BMI Distribution Comparison (Male vs Female)', color='white', fontsize=16, weight='light')
    sns.kdeplot(male_bmi, fill=True, color=WUWA_PALETTE[0], label='Male', alpha=0.2, linewidth=2)
    sns.kdeplot(female_bmi, fill=True, color=WUWA_PALETTE[2], label='Female', alpha=0.2, linewidth=2)
    plt.legend(facecolor='#121212', edgecolor='white')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig('parametric_bmi_comparison.png', dpi=150)
    plt.show()

    # Part III Nonparametric Test
def perform_nonparametric_test(df):
    """
    Nonparametric Test: Mann-Whitney U for Charges (Smoker vs Non-Smoker).
    Justification: Charges are skewed (not normal).
    """
    print("\n" + "="*60)
    print("Nonparametric Test Results (Charges by Smoker)")
    print("="*60)

    # Segregate groups
    if 'smoker' not in df.columns or 'charges' not in df.columns:
        print("Error: Required columns for nonparametric test not found.")
        return
    smoker_charges = df[df['smoker'] == 'yes']['charges']
    nonsmoker_charges = df[df['smoker'] == 'no']['charges']
    if smoker_charges.empty or nonsmoker_charges.empty:
        print("Error: One or both groups are empty for nonparametric test.")
        return
    # Run Mann-Whitney U Test
    u_stat, p_value = stats.mannwhitneyu(smoker_charges, nonsmoker_charges)
    print(f"U-Statistic: {u_stat:.4f}")
    print(f"P-Value:     {p_value:.4e}")
    print(f"Smoker Median:     ${smoker_charges.median():,.2f}")
    print(f"Non-Smoker Median: ${nonsmoker_charges.median():,.2f}")
    if p_value < 0.05:
        print("Result: Significant Difference (Reject Null)")
    else:
        print("Result: No Significant Difference (Fail to Reject Null)")
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.title('Charges Distribution (Smoker vs Non-Smoker)', color='white', fontsize=16, weight='light')
    sns.kdeplot(smoker_charges, fill=True, color=WUWA_PALETTE[2], label='Smoker', alpha=0.2, linewidth=2)
    sns.kdeplot(nonsmoker_charges, fill=True, color=WUWA_PALETTE[3], label='Non-Smoker', alpha=0.2, linewidth=2)
    plt.legend(facecolor='#121212', edgecolor='white')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig('nonparametric_charges_comparison.png', dpi=150)
    plt.show()
# ----------------------------------------------------------------------------------
# Main Execution Block
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load Data
    dataset_filename = 'Health Insurance Dataset.csv' 
    df = load_data(dataset_filename)
    
    if df is not None:
        # Defines Variables
        selected_quant = ['bmi', 'charges']
        selected_qual = ['smoker', 'region']

        # Runs Exploratory Analysis (Part I)
        plot_univariate_distributions(df, selected_quant, selected_qual)
        plot_bivariate_relationships(df)
        generate_descriptive_statistics(df)
        
# ----------------------------------------------------------------------------------
# Parametric Test & Nonparametric Test Execution Part II & III
# ----------------------------------------------------------------------------------
        
        # Parametric Test (T-Test for BMI)
        perform_parametric_test(df)

        # Nonparametric Test (Mann-Whitney U for Charges)
        perform_nonparametric_test(df)