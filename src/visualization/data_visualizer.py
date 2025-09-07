import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df, output_path):
    corr = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
