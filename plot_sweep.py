import matplotlib.pyplot as plt
import pandas as pd

def plot_sweep_results():
    try:
        # Read the sweep data
        df = pd.read_csv('sweep_results.csv')
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # FWHM vs Slit Width
        axes[0].plot(df['SlitWidth'], df['FWHM'], 'b-o', linewidth=2, markersize=8)
        axes[0].set_ylabel('FWHM', fontsize=12)
        axes[0].set_title('Interference Pattern Metrics vs Slit Width', fontsize=14)
        axes[0].grid(True, alpha=0.7)
        axes[0].fill_between(df['SlitWidth'], df['FWHM'], alpha=0.2)
        
        # Contrast vs Slit Width
        axes[1].plot(df['SlitWidth'], df['Contrast'], 'g-o', linewidth=2, markersize=8)
        axes[1].set_ylabel('Contrast', fontsize=12)
        axes[1].grid(True, alpha=0.7)
        axes[1].fill_between(df['SlitWidth'], df['Contrast'], alpha=0.2, color='green')
        
        # Symmetry vs Slit Width
        axes[2].plot(df['SlitWidth'], df['Symmetry'], 'r-o', linewidth=2, markersize=8)
        axes[2].set_xlabel('Slit Width (units)', fontsize=12)
        axes[2].set_ylabel('Symmetry', fontsize=12)
        axes[2].grid(True, alpha=0.7)
        axes[2].fill_between(df['SlitWidth'], df['Symmetry'], alpha=0.2, color='red')
        
        plt.tight_layout()
        plt.savefig('sweep_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'sweep_plot.png'")
        
        # Print summary
        print("\n=== Sweep Summary ===")
        print(df.to_string(index=False))
        print(f"\nFWHM Range: {df['FWHM'].min():.4f} - {df['FWHM'].max():.4f}")
        print(f"Contrast Range: {df['Contrast'].min():.4f} - {df['Contrast'].max():.4f}")
        print(f"Symmetry Range: {df['Symmetry'].min():.4f} - {df['Symmetry'].max():.4f}")
        
    except FileNotFoundError:
        print("Error: 'sweep_results.csv' not found. Run the sweep first (press 'S' in simulation).")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_sweep_results()
