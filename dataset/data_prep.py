import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TennisTrajectoryDataset(Dataset):
    def __init__(self, file_path, seq_length=20): #fixed sequence length for approximate episode length
        self.data = pd.read_csv(file_path, header=None)[1:]
        self.seq_length = seq_length

        # Column indices based on data structure
        self.columns = {
            'frame': 0,
            'ball_x': 1,
            'ball_y': 2,
            'velocity_x': 3,
            'velocity_y': 4,
            'shot_type': 5,
            'distance_travelled': 6
        }

        # Normalize features
        self._normalize_data()
        self._create_sequences()

    def _normalize_data(self):
        """Normalize features to zero mean and unit variance"""
        self.scalers = {}

        for col in ['ball_x', 'ball_y', 'velocity_x', 'velocity_y', 'distance_travelled']:
            mean = self.data[self.columns[col]].astype(float).mean()
            std = self.data[self.columns[col]].astype(float).std()
            self.data[self.columns[col]] = (self.data[self.columns[col]].astype(float) - mean) / (std + 1e-8)
            self.scalers[col] = (mean, std)

    def _create_sequences(self):
        """Create input-output sequences with temporal dependencies"""
        sequences = []
        data_size = len(self.data)

        for i in range(data_size - self.seq_length):
            seq = self.data.iloc[i:i+self.seq_length]
            input_seq = seq.iloc[:-1]
            target_seq = seq.iloc[1:]

            # Convert to state-action pairs
            states = []
            actions = []
            for j in range(len(input_seq)):
                state = self._get_state(input_seq.iloc[j])
                action = self._get_action(input_seq.iloc[j], target_seq.iloc[j])
                states.append(state)
                actions.append(action)

            sequences.append({
                'states': torch.FloatTensor(np.array(states)),
                'actions': torch.FloatTensor(np.array(actions))
            })

        self.sequences = sequences

    def _get_state(self, row):
        """Create state representation"""
        return np.array([
            float(row[self.columns['ball_x']]),
            float(row[self.columns['ball_y']]),
            float(row[self.columns['velocity_x']]),
            float(row[self.columns['velocity_y']]),
            float(row[self.columns['distance_travelled']]),
            float(row[self.columns['shot_type']])
        ])

    def _get_action(self, current_row, next_row):
        """Derive action from state transition"""
        dx = float(next_row[self.columns['ball_x']] - current_row[self.columns['ball_x']])
        dy = float(next_row[self.columns['ball_y']] - current_row[self.columns['ball_y']])
        return np.array([dx, dy])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]