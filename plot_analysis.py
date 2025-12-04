import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_interference_pattern():
    try:
        # Read the CSV data
        df = pd.read_csv('analysis_results.csv')
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(df['Bin_Y'], df['Counts'], color='blue', linewidth=1.5, label='Particle Hits')
        plt.fill_between(df['Bin_Y'], df['Counts'], color='blue', alpha=0.1)
        
        # Formatting
        plt.title('Quantum Time Interference Pattern (1 Million Particles)', fontsize=16)
        plt.xlabel('Screen Position (Y)', fontsize=12)
        plt.ylabel('Intensity (Counts)', fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Add some statistical annotations if available
        try:
            with open('results/analysis_summary.txt', 'r') as f:
                stats = f.read()
                # Place the stats box outside the plot, to the right
                plt.gcf().text(0.87, 0.5, stats, fontsize=10, verticalalignment='center',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        except FileNotFoundError:
            pass

        # Save the plot
        plt.savefig('interference_pattern.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'interference_pattern.png'")
        
    except FileNotFoundError:
        print("Error: 'analysis_results.csv' not found. Run the simulation analysis first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_interference_pattern()
