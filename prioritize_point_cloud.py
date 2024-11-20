import numpy as np
import heapq
import matplotlib.pyplot as plt

# Generate Mock Point Cloud Data (3D voxel grid)
np.random.seed(42)
point_cloud = np.random.rand(100, 3)  # 100 points with (x, y, z) coordinates

# Simulated Importance Scores (randomized)
importance_scores = np.random.rand(100)

# Combine data and scores
point_cloud_data = [{"point": point, "score": score} for point, score in zip(point_cloud, importance_scores)]


# Prioritize based on importance score
def prioritize_points(data, top_n=10):
    priority_queue = []
    for entry in data:
        heapq.heappush(priority_queue, (-entry["score"], entry["point"]))  # Max-heap by negative score

    prioritized_points = [heapq.heappop(priority_queue)[1] for _ in range(top_n)]
    return np.array(prioritized_points)


# Select top 10 points
top_points = prioritize_points(point_cloud_data)


# Visualization
def visualize_point_cloud(original, prioritized):
    fig = plt.figure(figsize=(10, 5))

    # Original Point Cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', alpha=0.5)
    ax1.set_title("Original Point Cloud")

    # Prioritized Points
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(prioritized[:, 0], prioritized[:, 1], prioritized[:, 2], c='red')
    ax2.set_title("Prioritized Points (Top 10)")
    plt.savefig("prioritized_point_cloud.png")
    plt.show()


# Visualize results
visualize_point_cloud(point_cloud, top_points)
