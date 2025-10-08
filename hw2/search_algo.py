import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Setup
start = np.array([1.0, 1.0])
goal = np.array([9.0, 9.0])
obstacle_min = np.array([4.0, 4.0])
obstacle_max = np.array([6.0, 6.0])
step_size = 1.5
goal_bias = 0.10
collision_resolution = 0.25

# Initialize tree with start node
tree_nodes = [start]
tree_edges = []

def sample_point():
    if np.random.uniform(0, 1) < goal_bias:
        return goal.copy(), "goal-biased"
    else:
        # Sample uniformly, avoiding obstacle interior
        while True:
            point = np.random.uniform(low=0, high=10, size=2)
            # Check if inside obstacle (closed set, so boundary counts as collision)
            if not (obstacle_min[0] <= point[0] <= obstacle_max[0] and 
                    obstacle_min[1] <= point[1] <= obstacle_max[1]):
                return point, "uniform"

def find_nearest(nodes, point):
    """Find nearest node in tree to given point"""
    min_dist = float('inf')
    nearest_idx = 0
    for i, node in enumerate(nodes):
        dist = np.linalg.norm(node - point)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
    return nearest_idx, nodes[nearest_idx]

def compute_new_node(q_near, q_rand, step_size):
    """Compute q_new by stepping from q_near toward q_rand"""
    direction = q_rand - q_near
    dist = np.linalg.norm(direction)
    
    if dist <= step_size:
        # q_rand is closer than step_size, so q_new = q_rand
        return q_rand.copy()
    else:
        # Step by step_size in direction of q_rand
        unit_direction = direction / dist
        return q_near + step_size * unit_direction

def check_collision(q_near, q_new, resolution):
    """Check collision along segment from q_near to q_new"""
    # Sample points along the segment
    dist = np.linalg.norm(q_new - q_near)
    num_checks = int(np.ceil(dist / resolution))
    
    for i in range(num_checks + 1):
        t = i / num_checks if num_checks > 0 else 0
        point = q_near + t * (q_new - q_near)
        
        # Check if point collides with obstacle (closed set, boundary counts)
        if (obstacle_min[0] <= point[0] <= obstacle_max[0] and 
            obstacle_min[1] <= point[1] <= obstacle_max[1]):
            return True  # Collision detected
    
    return False  # No collision

# Run 10 iterations
results = []

for iteration in range(1, 11):
    q_rand, sample_type = sample_point()
    nearest_idx, q_near = find_nearest(tree_nodes, q_rand)
    q_new = compute_new_node(q_near, q_rand, step_size)
    collision = check_collision(q_near, q_new, collision_resolution)
    
    edge_added = False
    if not collision:
        tree_nodes.append(q_new)
        tree_edges.append((q_near.copy(), q_new.copy()))
        edge_added = True
    
    # Store results
    results.append({
        'iteration': iteration,
        'q_rand': q_rand.copy(),
        'sample_type': sample_type,
        'q_near': q_near.copy(),
        'q_new': q_new.copy(),
        'collision': collision,
        'edge_added': edge_added
    })

# Print results table
print("RRT Growth Results (10 iterations)")
print("="*100)
print(f"{'Iter':<5} {'Sample':<12} {'q_rand':<20} {'q_near':<20} {'q_new':<20} {'Collision':<10} {'Added':<6}")
print("="*100)

for r in results:
    q_rand_str = f"({r['q_rand'][0]:.2f}, {r['q_rand'][1]:.2f})"
    q_near_str = f"({r['q_near'][0]:.2f}, {r['q_near'][1]:.2f})"
    q_new_str = f"({r['q_new'][0]:.2f}, {r['q_new'][1]:.2f})"
    
    print(f"{r['iteration']:<5} {r['sample_type']:<12} {q_rand_str:<20} {q_near_str:<20} {q_new_str:<20} {str(r['collision']):<10} {str(r['edge_added']):<6}")

print("\n" + "="*100)
print(f"Final tree size: {len(tree_nodes)} nodes, {len(tree_edges)} edges")