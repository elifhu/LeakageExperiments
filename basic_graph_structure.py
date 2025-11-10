import numpy as np
import matplotlib.pyplot as plt


def visualize_basic_concept():
    """
    Show the most basic graph concept
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Example 1: Simple friendship network
    ax = axes[0]
    ax.set_title("Example 1: Friendship Network", fontsize=14, fontweight='bold')

    # Positions of people
    pos = {
        'Alice': (0, 1),
        'Bob': (1, 1),
        'Carol': (0.5, 0)
    }

    # Friendships (edges)
    friendships = [
        ('Alice', 'Bob'),
        ('Bob', 'Carol'),
        ('Alice', 'Carol')
    ]

    # Draw friendships
    for person1, person2 in friendships:
        x_coords = [pos[person1][0], pos[person2][0]]
        y_coords = [pos[person1][1], pos[person2][1]]
        ax.plot(x_coords, y_coords, 'gray', linewidth=3, alpha=0.5, zorder=1)

    # Draw people
    for person, (x, y) in pos.items():
        ax.scatter(x, y, s=1500, c='lightblue', edgecolors='black', linewidth=2, zorder=2)
        ax.text(x, y, person, ha='center', va='center', fontsize=11, fontweight='bold')

    ax.text(0.5, -0.3, "Circles = People (Nodes)\nLines = Friendships (Edges)",
            ha='center', fontsize=10, style='italic')
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.5, 1.3)
    ax.axis('off')

    # Example 2: Road network
    ax = axes[1]
    ax.set_title("Example 2: Road Network", fontsize=14, fontweight='bold')

    cities_pos = {
        'NYC': (1, 1),
        'Boston': (1.5, 1.5),
        'Philly': (0.8, 0.5),
        'DC': (0.5, 0)
    }

    roads = [
        ('NYC', 'Boston'),
        ('NYC', 'Philly'),
        ('Philly', 'DC'),
        ('NYC', 'DC')
    ]

    # Draw roads
    for city1, city2 in roads:
        x_coords = [cities_pos[city1][0], cities_pos[city2][0]]
        y_coords = [cities_pos[city1][1], cities_pos[city2][1]]
        ax.plot(x_coords, y_coords, 'brown', linewidth=3, alpha=0.5, zorder=1)

    # Draw cities
    for city, (x, y) in cities_pos.items():
        ax.scatter(x, y, s=1500, c='lightcoral', edgecolors='black', linewidth=2, zorder=2)
        ax.text(x, y, city, ha='center', va='center', fontsize=10, fontweight='bold')

    ax.text(1, -0.3, "Circles = Cities (Nodes)\nLines = Roads (Edges)",
            ha='center', fontsize=10, style='italic')
    ax.set_xlim(0.2, 1.8)
    ax.set_ylim(-0.5, 1.8)
    ax.axis('off')

    # Example 3: Molecule
    ax = axes[2]
    ax.set_title("Example 3: Water Molecule (H₂O)", fontsize=14, fontweight='bold')

    atoms_pos = {
        'O': (0.5, 0.5),
        'H1': (0, 0),
        'H2': (1, 0)
    }

    bonds = [
        ('O', 'H1'),
        ('O', 'H2')
    ]

    # Draw bonds
    for atom1, atom2 in bonds:
        x_coords = [atoms_pos[atom1][0], atoms_pos[atom2][0]]
        y_coords = [atoms_pos[atom1][1], atoms_pos[atom2][1]]
        ax.plot(x_coords, y_coords, 'purple', linewidth=4, alpha=0.5, zorder=1)

    # Draw atoms
    colors = {'O': 'red', 'H1': 'lightblue', 'H2': 'lightblue'}
    for atom, (x, y) in atoms_pos.items():
        ax.scatter(x, y, s=1500, c=colors[atom], edgecolors='black', linewidth=2, zorder=2)
        label = 'O' if atom == 'O' else 'H'
        ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold')

    ax.text(0.5, -0.3, "Circles = Atoms (Nodes)\nLines = Bonds (Edges)",
            ha='center', fontsize=10, style='italic')
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.5, 0.8)
    ax.axis('off')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("WHAT IS A GRAPH?")
    print("=" * 70)
    print("\nA graph is made of two things:\n")
    print("  1. NODES (also called vertices) - the 'things'")
    print("  2. EDGES (also called links) - the 'relationships'\n")

    print("Creating visualization with 3 examples:")
    print("  • Friendship Network (people connected by friendships)")
    print("  • Road Network (cities connected by roads)")
    print("  • Water Molecule (atoms connected by bonds)")
    print()

    fig = visualize_basic_concept()
    plt.savefig('graph_basic_concept.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved as 'graph_basic_concept.png'")
    print("\nDisplaying the plot...")
    plt.show()