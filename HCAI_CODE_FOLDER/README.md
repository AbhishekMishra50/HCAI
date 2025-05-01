## Reward Model Training and Worker Embedding Clustering

### Steps to Run the Code

**1. Create an Environment**

Create a Conda or Python environment and install the required libraries from requirement.txt file. Use the following commands.
```
conda create -n <ENV_NAME> python=3.10
pip install -r requirements.txt
```


**2. Train the Reward Model**

After installing the required libraries, you can run the Python file for training the reward model using the command:
```
python train_reward_model_gptneo.py
```
**3. Enable Logging**

We are storing our logs in Weights & Biases, so create your account using [Weights and Biases](https://wandb.ai/site/). Set your API key to enable logging.

**4. Run Experiments**
In the _train_reward_model_gptneo.py_ file, you can change the **worker_embedding_dim**. For our experiments, we used the values: 8, 16, and 32. After training, the final model checkpoint will be saved in the output directory
```
rm_checkpoint/
```

**4. Visualize Worker Embeddings**

After training and saving the checkpoint, run the notebook _worker_ids.ipynb_ to visualise the similarities between workers and create clusters based on worker embeddings.


**Important**: Change the path variable to your actual path where the checkpoints are saved.
