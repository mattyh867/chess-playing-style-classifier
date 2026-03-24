import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


class DatasetExplorer: 
    def __init__(self, csv_path: str):
        """Load dataset"""
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} games from {csv_path}")
    
    def print_summary(self):
        """Print dataset summary"""
        print("\nDATASET SUMMARY")
        
        print(f"\nTotal entries: {len(self.df)}")
        print(f"Unique players: {self.df['player_name'].nunique()}")
        print(f"Unique games: {self.df['game_id'].nunique()}")
        
        print("\nPLAYSTYLE DISTRIBUTION")
        print(self.df['label'].value_counts())
        print("\nPercentages:")
        for style, count in self.df['label'].value_counts(normalize=True).items():
            print(f"  {style}: {count*100:.1f}%")
        
        print("\nRATING STATISTICS")
        print(f"Mean:   {self.df['player_elo'].mean():.0f}")
        print(f"Median: {self.df['player_elo'].median():.0f}")
        print(f"Std:    {self.df['player_elo'].std():.0f}")
        print(f"Min:    {self.df['player_elo'].min():.0f}")
        print(f"Max:    {self.df['player_elo'].max():.0f}")
        
        print("\nPLAYSTYLE BY RATING")
        for style in self.df['label'].unique():
            style_df = self.df[self.df['label'] == style]
            print(f"{style:12s}: {style_df['player_elo'].mean():.0f} ± {style_df['player_elo'].std():.0f}")
        
        print("\nACCURACY STATISTICS")
        print(f"Mean CP Loss: {self.df['avg_centipawn_loss'].mean():.1f}")
        print(f"Mean Accuracy: {self.df['accuracy'].mean()*100:.1f}%")
        
        print("\nPLAYSTYLE CHARACTERISTICS (Averages)")
        
        metrics = ['checks_per_move', 'captures_per_move', 'sacrifices', 
                  'early_attacks', 'prophylactic_moves', 'simplifications']
        
        for style in self.df['label'].unique():
            print(f"\n{style.upper()}:")
            style_df = self.df[self.df['label'] == style]
            for metric in metrics:
                if metric in style_df.columns:
                    print(f"  {metric:25s}: {style_df[metric].mean():.3f}")
        

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Explore labeled chess dataset')
    parser.add_argument('--csv', required=True, help='Path to labeled CSV file')
    
    args = parser.parse_args()
    
    explorer = DatasetExplorer(args.csv)
    explorer.print_summary()

if __name__ == "__main__":
    main()