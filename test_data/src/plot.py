import matplotlib.pyplot as plt

def plot_trajectories(trajectories, title):
    plt.figure(figsize=(8, 6))
    for i in range(trajectories.shape[0]):
        plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], alpha=0.3, color='blue')
    
    plt.scatter(trajectories[:, 0, 0], trajectories[:, 0, 1], color='red', label='Start')
    plt.scatter(trajectories[:, -1, 0], trajectories[:, -1, 1], color='green', s=10, label='End')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

plot_trajectories(trajectories, "Linear 2D System Trajectories")