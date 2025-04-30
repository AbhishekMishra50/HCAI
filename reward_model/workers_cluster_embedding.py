import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
from reward_model import GPTRewardModel

def main():
    checkpoint_path = "/users/student/pg/pg24/abhishek/rm_checkpoint_multi/final_best_model"
    base_model = "EleutherAI/gpt-neo-1.3B"
    num_workers = 21
    embedding_dim = 8
    num_clusters = 1

    # Load model and extract worker embeddings
    model = GPTRewardModel(base_model, num_workers=num_workers, worker_embedding_dim=embedding_dim)
    checkpoint = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    embeddings = model.worker_embeddings.weight.detach().cpu().numpy()

    # KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # t-SNE for 2D visualization
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap="tab10")
    for i, (x, y) in enumerate(embeddings_2d):
        plt.text(x, y, str(i), fontsize=8)
    plt.title("Worker Embeddings Clustering")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.colorbar(scatter, label="Cluster ID")
    plt.savefig("worker_clusters1.png", dpi=300)
    plt.show()

    # Save results to CSV
    df = pd.DataFrame({
        "worker_id": list(range(num_workers)),
        "cluster": cluster_labels,
        "tsne_x": embeddings_2d[:, 0],
        "tsne_y": embeddings_2d[:, 1]
    })
    df.to_csv("worker_clusters.csv", index=False)

if __name__ == "__main__":
    main()
