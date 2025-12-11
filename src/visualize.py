import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def plot_steering_distribution(csv_path, output_path='logs/training_plots/steering_distribution.png'):
    """
    Plot histogram of steering angle distribution
    """
    # Read CSV
    df = pd.read_csv(csv_path, header=None)
    steering_angles = df[1].values
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot histogram
    n, bins, patches = plt.hist(steering_angles, bins=31, edgecolor='black', alpha=0.7)
    
    # Calculate statistics
    mean_angle = np.mean(steering_angles)
    std_angle = np.std(steering_angles)
    median_angle = np.median(steering_angles)
    
    # Add vertical lines for mean and median
    plt.axvline(mean_angle, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_angle:.4f}')
    plt.axvline(median_angle, color='green', linestyle='--', linewidth=2, label=f'Median: {median_angle:.4f}')
    
    # Labels and title
    plt.xlabel('Steering Angle', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Steering Angle Distribution\nTotal Samples: {len(steering_angles)} | Std: {std_angle:.4f}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved steering distribution plot to: {output_path}")
    
    # Show statistics
    print("\n=== Steering Angle Statistics ===")
    print(f"Total samples: {len(steering_angles)}")
    print(f"Mean: {mean_angle:.6f}")
    print(f"Median: {median_angle:.6f}")
    print(f"Std Dev: {std_angle:.6f}")
    print(f"Min: {np.min(steering_angles):.6f}")
    print(f"Max: {np.max(steering_angles):.6f}")
    print(f"Range: {np.max(steering_angles) - np.min(steering_angles):.6f}")
    
    # Check balance
    zero_count = np.sum(np.abs(steering_angles) < 0.01)
    zero_percentage = (zero_count / len(steering_angles)) * 100
    print(f"\nZero steering (~0): {zero_count} ({zero_percentage:.2f}%)")
    
    if zero_percentage > 50:
        print("⚠️  WARNING: Dataset is heavily biased toward straight driving!")
        print("   Consider running: python -m src.utils --balance")
    else:
        print("✓ Dataset appears reasonably balanced")
    
    plt.show()


def compare_distributions(csv1, csv2, output_path='logs/training_plots/distribution_comparison.png'):
    """
    Compare steering distributions before and after balancing
    """
    df1 = pd.read_csv(csv1, header=None)
    df2 = pd.read_csv(csv2, header=None)
    
    angles1 = df1[1].values
    angles2 = df2[1].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before balancing
    ax1.hist(angles1, bins=31, edgecolor='black', alpha=0.7, color='coral')
    ax1.set_xlabel('Steering Angle')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Before Balancing\nSamples: {len(angles1)}')
    ax1.grid(True, alpha=0.3)
    
    # After balancing
    ax2.hist(angles2, bins=31, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.set_xlabel('Steering Angle')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'After Balancing\nSamples: {len(angles2)}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved comparison plot to: {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize steering angle distribution')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--compare', default='', help='Path to second CSV for comparison')
    parser.add_argument('--output', default='logs/training_plots/steering_distribution.png')
    args = parser.parse_args()
    
    if args.compare:
        compare_distributions(args.csv, args.compare)
    else:
        plot_steering_distribution(args.csv, args.output)