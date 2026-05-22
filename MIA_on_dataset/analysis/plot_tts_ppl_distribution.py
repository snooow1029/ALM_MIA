#!/usr/bin/env python3
"""
Plot PPL distribution for different TTS methods
Shows distribution for tts, tts_61, tts_460, and original
with different colors for is_member=1 and is_member=0
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_ppl_data(json_file_path):
    """
    Load PPL values from JSON file
    
    Returns:
        dict: {
            'member': [ppl_values for is_member=1],
            'non_member': [ppl_values for is_member=0]
        }
    """
    if not os.path.exists(json_file_path):
        print(f"Warning: File not found: {json_file_path}")
        return {'member': [], 'non_member': []}
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    member_ppls = []
    non_member_ppls = []
    
    if 'samples' not in data:
        print(f"Warning: No 'samples' key in {json_file_path}")
        return {'member': [], 'non_member': []}
    
    for sample_id, sample_data in data['samples'].items():
        # Get is_member value
        is_member = sample_data.get('is_member', None)
        
        # Get PPL value
        if 'metrics' in sample_data and 'ppl' in sample_data['metrics']:
            ppl_value = sample_data['metrics']['ppl'].get('raw_value', None)
            
            if ppl_value is not None:
                if is_member == 1:
                    member_ppls.append(ppl_value)
                elif is_member == 0:
                    non_member_ppls.append(ppl_value)
    
    print(f"Loaded {json_file_path}:")
    print(f"  Member samples: {len(member_ppls)}")
    print(f"  Non-member samples: {len(non_member_ppls)}")
    
    return {
        'member': member_ppls,
        'non_member': non_member_ppls
    }

def plot_ppl_distribution(base_dir, output_file):
    """
    Plot PPL distribution for different TTS methods
    
    Args:
        base_dir: Base directory containing tts, tts_61, tts_460, original subdirectories
        output_file: Path to save the output figure
    """
    # Define TTS types and their display names
    tts_types = {
        'tts': 'TTS',
        'tts_61': 'TTS_61',
        'tts_460': 'TTS_460',
        'original': 'Original'
    }
    
    # Define 8 distinct colors for 4 TTS types × 2 membership types
    # Format: (tts_type, is_member) -> color
    colors = {
        ('tts', 1): '#FF6B6B',        # Red - TTS Member
        ('tts', 0): '#FFB6C1',        # Light Pink - TTS Non-member
        ('tts_61', 1): '#4ECDC4',     # Teal - TTS_61 Member
        ('tts_61', 0): '#95E1D3',     # Light Teal - TTS_61 Non-member
        ('tts_460', 1): '#45B7D1',    # Blue - TTS_460 Member
        ('tts_460', 0): '#96CEB4',    # Light Green - TTS_460 Non-member
        ('original', 1): '#FFA07A',   # Light Salmon - Original Member
        ('original', 0): '#F0E68C',   # Khaki - Original Non-member
    }
    
    # Load data for each TTS type
    all_data = {}
    for tts_type, display_name in tts_types.items():
        json_file = os.path.join(base_dir, tts_type, 'two_stage_audio_mia_results_original.json')
        all_data[tts_type] = load_ppl_data(json_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PPL Distribution Comparison: Different TTS Methods', fontsize=16, fontweight='bold')
    
    # Plot each TTS type
    for idx, (tts_type, display_name) in enumerate(tts_types.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        data = all_data[tts_type]
        member_ppls = data['member']
        non_member_ppls = data['non_member']
        
        # Plot histograms with distinct colors
        if member_ppls:
            ax.hist(member_ppls, bins=50, alpha=0.7, label=f'Member (n={len(member_ppls)})', 
                   color=colors[(tts_type, 1)], edgecolor='black', linewidth=0.5)
        
        if non_member_ppls:
            ax.hist(non_member_ppls, bins=50, alpha=0.7, label=f'Non-member (n={len(non_member_ppls)})', 
                   color=colors[(tts_type, 0)], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('PPL (Perplexity)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{display_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        if member_ppls and non_member_ppls:
            member_mean = np.mean(member_ppls)
            member_std = np.std(member_ppls)
            non_member_mean = np.mean(non_member_ppls)
            non_member_std = np.std(non_member_ppls)
            
            stats_text = f'Member: μ={member_mean:.3f}, σ={member_std:.3f}\n'
            stats_text += f'Non-member: μ={non_member_mean:.3f}, σ={non_member_std:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    
    # Create combined comparison plot with all 8 distributions
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    
    # Plot all 8 distributions together
    for tts_type, display_name in tts_types.items():
        data = all_data[tts_type]
        member_ppls = data['member']
        non_member_ppls = data['non_member']
        
        if member_ppls:
            ax2.hist(member_ppls, bins=50, alpha=0.6, 
                    label=f'{display_name} - Member (n={len(member_ppls)})', 
                    color=colors[(tts_type, 1)], 
                    edgecolor='black', linewidth=0.3)
        
        if non_member_ppls:
            ax2.hist(non_member_ppls, bins=50, alpha=0.6, 
                    label=f'{display_name} - Non-member (n={len(non_member_ppls)})', 
                    color=colors[(tts_type, 0)], 
                    edgecolor='black', linewidth=0.3)
    
    ax2.set_xlabel('PPL (Perplexity)', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.set_title('PPL Distribution: All TTS Methods Combined (8 Distributions)', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=10, ncol=2, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_output = output_file.replace('.png', '_combined.png')
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    print(f"Combined figure saved to: {combined_output}")
    
    # Create violin plot for better comparison
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    # Prepare data for violin plot
    plot_data = []
    plot_labels = []
    plot_colors_list = []
    
    for tts_type, display_name in tts_types.items():
        data = all_data[tts_type]
        member_ppls = data['member']
        non_member_ppls = data['non_member']
        
        if member_ppls:
            plot_data.append(member_ppls)
            plot_labels.append(f'{display_name}\nMember')
            plot_colors_list.append(colors[(tts_type, 1)])
        
        if non_member_ppls:
            plot_data.append(non_member_ppls)
            plot_labels.append(f'{display_name}\nNon-member')
            plot_colors_list.append(colors[(tts_type, 0)])
    
    if plot_data:
        parts = ax3.violinplot(plot_data, positions=range(len(plot_data)), 
                               showmeans=True, showmedians=True)
        
        # Color the violins
        for pc, color in zip(parts['bodies'], plot_colors_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax3.set_xticks(range(len(plot_labels)))
        ax3.set_xticklabels(plot_labels, rotation=45, ha='right')
        ax3.set_ylabel('PPL (Perplexity)', fontsize=14)
        ax3.set_title('PPL Distribution: Violin Plot Comparison', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    violin_output = output_file.replace('.png', '_violin.png')
    plt.savefig(violin_output, dpi=300, bbox_inches='tight')
    print(f"Violin plot saved to: {violin_output}")

if __name__ == '__main__':
    import argparse

    mia_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_base_dir = os.path.join(mia_root, 'two_stage_mia_results', 'spgispeech', 'prompt_a')
    
    parser = argparse.ArgumentParser(description='Plot PPL distribution for different TTS methods')
    parser.add_argument('--base-dir', type=str, 
                       default=default_base_dir,
                       help='Base directory containing TTS subdirectories')
    parser.add_argument('--output', type=str,
                       default=os.path.join(default_base_dir, 'analysis', 'tts_ppl_distribution.png'),
                       help='Output file path for the figure')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    plot_ppl_distribution(args.base_dir, args.output)
