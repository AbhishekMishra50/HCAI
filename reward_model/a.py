import torch


checkpoint = torch.load('/users/student/pg/pg24/abhishek/rm_checkpoint_multi/final_best_model/pytorch_model.bin', map_location='cpu') 

worker_embeddings = checkpoint['worker_embeddings.weight']
print(worker_embeddings.shape)

print(worker_embeddings)


# worker_embeddings_np = worker_embeddings.cpu().numpy()
