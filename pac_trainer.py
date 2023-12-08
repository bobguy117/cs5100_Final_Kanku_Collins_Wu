"""
Introductory Deep Learning exercise for training agents to navigate
small Pacman Mazes
"""

import time
import random
import re
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from constants import *
from maze_gen import MazeGen
import matplotlib.pyplot as plt

class PacmanMazeDataset(Dataset):
    """
    PyTorch Dataset extension used to vectorize Pacman mazes consisting of the
    entities listed in Constants.py to be used for neural network training
    See: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """

    maze_entity_indexes = {entity: index for index, entity in enumerate(Constants.ENTITIES)}
    move_indexes = {move: index for index, move in enumerate(Constants.MOVES)}

    def __init__(self, training_data):
        self.training_data = training_data

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        row = self.training_data.iloc[idx]
        maze, move = row["X"], row["y"]
        print(f"Maze: {maze}")

        maze_tensor = self.vectorize_maze(maze)
        move_tensor = self.vectorize_move(move)

        return maze_tensor, move_tensor

    @staticmethod
    def vectorize_move(move):
        move_index = PacmanMazeDataset.move_indexes[move]
        move_tensor = torch.nn.functional.one_hot(
            torch.tensor(move_index, dtype=torch.long),
            num_classes=len(Constants.MOVES)
        ).float()
        flattened_tensor = torch.flatten(move_tensor)
        return flattened_tensor

    def vectorize_maze(self, maze):
        """
        Vectorizes a Pacman maze string into a 1-D tensor.

        Args:
            maze: A string representing the Pacman maze.

        Returns:
            A 1-D tensor containing the one-hot encoded representation of the maze.
        """
        rows = len(maze.split('\n'))
        cols = len(maze.split('\n')[0])
        maze_rows = maze.count('\n') + 1
        maze_cols = len(maze.split('\n')[0]) if maze_rows > 0 else 0
        #print(f"Original Maze Dimensions: {maze_rows} rows x {maze_cols} columns")
        maze_indices = [self.maze_entity_indexes.get(entity) for row in maze.split('\n') for entity in row]
    
        # Filter out None values
        maze_indices = [index for index in maze_indices if index is not None]

        maze_tensor = torch.tensor(maze_indices, dtype=torch.long)
        one_hot_encoded = torch.nn.functional.one_hot(
            maze_tensor, num_classes=len(Constants.ENTITIES)
        ).float()
        
        flattened_tensor = torch.flatten(one_hot_encoded)
        # Ensure that the flattened tensor size matches the expected input size in PacNet
        #print("Flattened Tensor size:", flattened_tensor.size())
        return flattened_tensor


class PacNet(nn.Module):
    """
    PyTorch Neural Network extension for the Pacman gridworld, which is fit to a
    particular maze configuration (walls, and optionally pellets, are in fixed spots)
    See: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
    """
    
    def __init__(self, input_size, output_size):
        """
        Initializes a PacNet for the given maze, which has maze-specific configuration
        requirements like the number of rows and cols. Used to perform imitation learning
        to select one of the 4 available actions in Constants.MOVES in response to the
        positional maze entities in Constants.ENTITIES
        :maze: The Pacman Maze structure on which this PacNet will be trained
        """
        super(PacNet, self).__init__()

        hidden_size1 = 128
        hidden_size2 = 64 
          
    
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        # Dropout layer for regularization 
        self.dropout = nn.Dropout(0.5)
        print("PacNet initialized.")

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        print("Input size:", x.size())
    
        x = F.relu(self.fc1(x))
        print("Size after fc1:", x.size())
    
        x = self.dropout(x)
    
        x = F.relu(self.fc2(x))
        print("Size after fc2:", x.size())
    
        x = self.dropout(x)
    
        logits = self.fc3(x)
        print("Output size:", logits.size())
    
        return logits



def train_loop(dataloader, model, loss_fn, optimizer, device):
    """
    PyTorch Neural Network optimization loop; need not be modified unless tweaks are
    desired.
    See: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    """
    size = len(dataloader.dataset)
    model.train()

    # Lists to store average loss values and batch indices
    average_losses = []
    batch_indices = []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current = batch * len(X)
        average_loss = loss.item() / len(X)
        print(f"Batch: {batch}, Average Loss: {average_loss:.4f}  [{current:>5d}/{size:>5d}]")

        # Store values for plotting
        average_losses.append(average_loss)
        batch_indices.append(batch)


    print("Training loop completed.")
    # Plot average loss after the entire training loop
    plt.plot(batch_indices, average_losses, label='Average Loss')
    plt.xlabel('Batch Index')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Over Batches')
    plt.legend()
    plt.show()

    

def main():
    # Step 1: Read the raw training data
    our_data_path = "dat\samples_from_all_workers_in_lobby_TQBD_461.csv" 
    generated_data_path = "dat/generated_data.csv"  

    our_data = pd.read_csv(our_data_path)
    generated_data = pd.read_csv(generated_data_path)

    # Step 2: Construct PacmanMazeDataset objects
    our_dataset = PacmanMazeDataset(our_data)
    generated_dataset = PacmanMazeDataset(generated_data)

    # Step 3: Create DataLoaders
    batch_size = 27  
    class_dataloader = DataLoader(our_dataset, batch_size=batch_size, shuffle=True)
    generated_dataloader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=True)

    # Step 4: Check data loading and vectorization
    sample_our_data = our_dataset.__getitem__(0)
    sample_generated_data = generated_dataset.__getitem__(0)

    # Ensure sample_our_data[0] represents a maze (tensor)
    # Ensure sample_our_data[0] represents a maze (tensor)
    assert isinstance(sample_our_data[0], torch.Tensor), "Expecting a torch.Tensor for the maze."
    assert isinstance(sample_generated_data[0], torch.Tensor)
    # Flatten the maze tensor
    flattened_maze = sample_our_data[0].view(1, -1)

    # Print the flattened maze size for debugging
    print("Flattened Tensor size:", flattened_maze.size())

    # initialize the nn using the flattened maze
    model = PacNet(flattened_maze.size(1), len(Constants.MOVES))
    # initialize the nn using the maze dimensions
    model = model.to(Constants.DEVICE)

    print("Sample our Data:")
    print("Vectorized Maze:", sample_our_data[0])
    print("Vectorized Move:", sample_our_data[1])

    print("\nSample Generated Data:")
    print("Vectorized Maze:", sample_generated_data[0])
    print("Vectorized Move:", sample_generated_data[1])

    # Optimization
    learning_rate = 1e-3
    epochs = 100
    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()
    # Define device before calling train_loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Call train_loop with the defined device
    train_loop(class_dataloader, model, loss_fn, optimizer, device)
    # Save the state_dict only
    torch.save(model.state_dict(), Constants.PARAM_PATH)


    # Alternatively, iterate through the DataLoader
    #for batch in class_dataloader:
        #print("Batch Vectorized Maze Shape:", batch[0].shape)
        #print("Batch Vectorized Move Shape:", batch[1].shape)
        #break  # Print only the first batch for brevity

    

if __name__ == "__main__":
    """
    Main method used to load training data, construct PacNet, and then
    train it, finally saving the network's parameters for use by the
    pacman agent.
    See: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    """
    main()
    
    # TODO: Task 4 Here
    
    # TODO: Task 5 Here
    
    # TODO: Task 6 Here
    
