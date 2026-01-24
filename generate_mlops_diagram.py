#!/usr/bin/env python3
"""
Generate MLOps Pipeline Architecture Diagram

This script generates a professional pipeline architecture diagram for the
Predictive Maintenance MLOps project.

Usage:
    python generate_mlops_diagram.py

Output:
    screenshots/mlops_pipeline_architecture.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# =============================================================================
# CONFIGURATION - Modify these values as needed
# =============================================================================

# Color palette
COLORS = {
    'data_pipeline': '#4B5563',      # Grey
    'model_development': '#D97706',  # Amber
    'deployment': '#2563EB',         # Blue
    'automation': '#059669',         # Green
    'text_dark': '#374151',
    'connector': '#9CA3AF',
    'background': '#FAFBFC',
}

# Content - Update these to match your project
CONTENT = {
    'title': 'MLOps Pipeline Architecture',
    'subtitle': 'End-to-End Automated Machine Learning Workflow',

    # Data Pipeline boxes
    'raw_dataset': ('Raw Dataset', 'engine_data.csv'),
    'data_registration': ('Data Registration', 'HF Dataset Hub'),
    'data_preparation': ('Data Preparation', 'Stratified Split (70/30)'),

    # Model Development boxes
    'model_comparison': ('Model Comparison', '5 Algorithms Tested'),
    'optimization': ('Bayesian Optimization', '30 Iterations'),
    'tracking': ('MLflow Tracking', 'Experiment Logging'),

    # Deployment boxes
    'model_registry': ('Model Registry', 'HF Model Hub'),
    'docker_build': ('Docker Build', 'Streamlit App'),
    'hf_spaces': ('HF Spaces', 'Production'),

    # Automation boxes
    'github_actions': ('GitHub Actions Workflow', 'Automated Pipeline Execution'),
    'monitoring': ('Continuous Monitoring', 'Performance Tracking'),
    'retraining': ('Model Retraining', 'Automated Updates'),

    # Connector labels
    'connector_1': 'Processed Data',
    'connector_2': 'Best Model',
}

# Output settings
OUTPUT_PATH = 'screenshots/mlops_pipeline_architecture.png'
DPI = 200


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

def draw_box(ax, x, y, w, h, color, title, subtitle, shadow=True):
    """Draw a professional rounded box with shadow"""
    if shadow:
        shadow_box = FancyBboxPatch(
            (x - w/2 + 0.05, y - h/2 - 0.05), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor='#00000015', edgecolor='none'
        )
        ax.add_patch(shadow_box)

    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=color, edgecolor='none'
    )
    ax.add_patch(box)
    ax.text(x, y + 0.18, title, fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(x, y - 0.18, subtitle, fontsize=9,
            ha='center', va='center', color='white', alpha=0.92)


def draw_connector_pill(ax, x, y, text, color='#374151'):
    """Draw a pill-shaped connector label"""
    box = FancyBboxPatch(
        (x - 0.65, y - 0.18), 1.3, 0.36,
        boxstyle="round,pad=0.01,rounding_size=0.15",
        facecolor='white', edgecolor='#D1D5DB', linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, y, text, fontsize=8, fontweight='medium',
            ha='center', va='center', color=color)


def draw_section_header(ax, x, y, num, title, color):
    """Draw section header with number badge"""
    circle = plt.Circle((x - 1.3, y), 0.22, facecolor=color, edgecolor='none')
    ax.add_patch(circle)
    ax.text(x - 1.3, y, num, fontsize=10, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(x - 0.9, y, title, fontsize=12, fontweight='bold',
            ha='left', va='center', color=color)


# =============================================================================
# MAIN DIAGRAM GENERATION
# =============================================================================

def generate_diagram():
    """Generate the MLOps pipeline architecture diagram"""

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    # Main title
    ax.text(9, 11.4, CONTENT['title'], fontsize=28, fontweight='bold',
            ha='center', va='center', color=COLORS['text_dark'])
    ax.text(9, 10.85, CONTENT['subtitle'], fontsize=13,
            ha='center', va='center', color='#6B7280', style='italic')

    # Divider line
    ax.plot([2, 16], [10.5, 10.5], color='#E5E7EB', linewidth=1.5)

    # Section headers
    draw_section_header(ax, 3.5, 9.9, '1', 'Data Pipeline', COLORS['data_pipeline'])
    draw_section_header(ax, 9, 9.9, '2', 'Model Development', COLORS['model_development'])
    draw_section_header(ax, 15.1, 9.9, '3', 'Deployment', COLORS['deployment'])

    # Data Pipeline boxes (Grey)
    draw_box(ax, 3.2, 9.0, 2.5, 1.0, COLORS['data_pipeline'], *CONTENT['raw_dataset'])
    draw_box(ax, 3.2, 7.7, 2.5, 1.0, COLORS['data_pipeline'], *CONTENT['data_registration'])
    draw_box(ax, 3.2, 6.4, 2.5, 1.0, COLORS['data_pipeline'], *CONTENT['data_preparation'])

    # Model Development boxes (Amber)
    draw_box(ax, 9, 9.0, 2.7, 1.0, COLORS['model_development'], *CONTENT['model_comparison'])
    draw_box(ax, 9, 7.7, 2.7, 1.0, COLORS['model_development'], *CONTENT['optimization'])
    draw_box(ax, 9, 6.4, 2.7, 1.0, COLORS['model_development'], *CONTENT['tracking'])

    # Deployment boxes (Blue)
    draw_box(ax, 14.8, 9.0, 2.5, 1.0, COLORS['deployment'], *CONTENT['model_registry'])
    draw_box(ax, 14.8, 7.7, 2.5, 1.0, COLORS['deployment'], *CONTENT['docker_build'])
    draw_box(ax, 14.8, 6.4, 2.5, 1.0, COLORS['deployment'], *CONTENT['hf_spaces'])

    # Connector pills
    draw_connector_pill(ax, 6.1, 6.4, CONTENT['connector_1'])
    draw_connector_pill(ax, 11.9, 6.4, CONTENT['connector_2'])

    # Vertical connectors within columns
    for x in [3.2, 9, 14.8]:
        ax.plot([x, x], [8.45, 8.25], color=COLORS['connector'], linewidth=2)
        ax.plot([x, x], [7.15, 6.95], color=COLORS['connector'], linewidth=2)

    # Horizontal connectors between columns
    ax.annotate('', xy=(5.4, 6.4), xytext=(4.5, 6.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['connector'], lw=2))
    ax.annotate('', xy=(7.6, 6.4), xytext=(6.8, 6.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['connector'], lw=2))
    ax.annotate('', xy=(11.2, 6.4), xytext=(10.4, 6.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['connector'], lw=2))
    ax.annotate('', xy=(13.5, 6.4), xytext=(12.6, 6.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['connector'], lw=2))

    # CI/CD Section header
    draw_section_header(ax, 9, 5.0, '4', 'CI/CD Orchestration', COLORS['automation'])

    # GitHub Actions box (Green)
    github_shadow = FancyBboxPatch(
        (2.7, 3.55), 12.6, 1.1,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#00000012', edgecolor='none'
    )
    ax.add_patch(github_shadow)

    github_box = FancyBboxPatch(
        (2.65, 3.6), 12.6, 1.1,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COLORS['automation'], edgecolor='none'
    )
    ax.add_patch(github_box)
    ax.text(9, 4.25, CONTENT['github_actions'][0], fontsize=13, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(9, 3.9, CONTENT['github_actions'][1], fontsize=10,
            ha='center', va='center', color='white', alpha=0.9)

    # Connectors from main flow to GitHub Actions
    for x in [3.2, 9, 14.8]:
        ax.plot([x, x], [5.9, 5.35], color=COLORS['connector'], linewidth=2,
                linestyle='--', alpha=0.7)
        ax.plot([x, x], [4.65, 4.75], color=COLORS['connector'], linewidth=2,
                linestyle='--', alpha=0.7)

    # Bottom boxes (Green)
    draw_box(ax, 5.5, 2.3, 3.0, 0.9, COLORS['automation'], *CONTENT['monitoring'])
    draw_box(ax, 12.5, 2.3, 3.0, 0.9, COLORS['automation'], *CONTENT['retraining'])

    # Connectors to bottom boxes
    ax.plot([5.5, 5.5], [3.6, 2.8], color=COLORS['connector'], linewidth=2,
            linestyle='--', alpha=0.7)
    ax.plot([12.5, 12.5], [3.6, 2.8], color=COLORS['connector'], linewidth=2,
            linestyle='--', alpha=0.7)

    # Legend
    ax.text(9, 1.45, 'Pipeline Components', fontsize=11, fontweight='bold',
            ha='center', color=COLORS['text_dark'])

    legend_items = [
        (COLORS['data_pipeline'], 'Data Processing'),
        (COLORS['model_development'], 'Model Training'),
        (COLORS['deployment'], 'Deployment'),
        (COLORS['automation'], 'Automation')
    ]

    start_x = 9 - (len(legend_items) * 3.2) / 2 + 0.8
    for i, (color, label) in enumerate(legend_items):
        x = start_x + i * 3.2
        box = FancyBboxPatch(
            (x, 0.85), 0.5, 0.35,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            facecolor=color, edgecolor='none'
        )
        ax.add_patch(box)
        ax.text(x + 0.7, 1.02, label, fontsize=10, ha='left', va='center',
                color=COLORS['text_dark'])

    # Save
    plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight',
                facecolor=COLORS['background'], edgecolor='none', pad_inches=0.3)
    plt.close()

    print(f"Created: {OUTPUT_PATH}")


if __name__ == '__main__':
    generate_diagram()
