Reward Model Training and Worker Embedding Clustering

Steps to Run the Code

STEP 1: Create an Environment
Create a Conda or Python environment (Python version ≥ 3.10).

STEP 2: Install Required Libraries
In the environment you have created, install the required libraries by running the command:
pip install -r requirements.txt

STEP 3: Train the Reward Model
After installing the required libraries, you can run the Python file for training the reward model using the command:
python train_reward_model_gptneo.py

We are storing our logs in Weights & Biases, so create your account using the following link:
https://wandb.ai/site/
Then set your API key to enable logging.

In the train_reward_model_gptneo.py file, you can change the worker_embedding_dim.
For our experiments, we used the values: 8, 16, and 32.

After training, the final model checkpoint will be saved in the output directory:
rm_checkpoint/

STEP 4: Visualize Worker Embeddings
After training and saving the checkpoint, run the notebook worker_ids.ipynb to visualize the similarities between workers and create clusters based on worker embeddings.

Important: Change the path variable to your actual path where the checkpoints are saved.

Results
We have a folder named Results/ which contains plots and clusters for different values of worker_embedding_dim used in our experiments.
