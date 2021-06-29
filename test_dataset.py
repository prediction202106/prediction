import os
import numpy as np
import random
import torch


class TestDataset(torch.utils.data.Dataset):
    '''
    'TRAJECTORY_FEATURE': 
        shape: [vehicle_num, time_steps, feature_dimension]
        feature definition: [x, y, heading, bool, s, l, timestamp, brake, signal], 
    'SL_TRAJECTORY_FEATURE':  
        shape: [vehicle_num * lane_num, time_steps, feature_dimension]
        feature definition: [s, l, lane_turn, bool, accumulated_s, lane_idx, lane_hash]
    'FORWARD_MAP_FEATURE':
        shape: [lane_num, time_steps, feature_dimension]
        feature definition: [dx, dy, lane_turn, bool, lane_hash]
    'NEIGHBOR_FEATURE'

    vehicle_num: total vehicle number in this frame, vehicle_num <= 10
    lane_num: total lane number in this frame, lane_num <= 10
    x, y: vehicle position in world coordinate system
    s, l: vehicle position in map coordinate system
    heading: vehicle heading in world coordinate system
    accumulate_s: distance travelled along current lane is (s - accumulated_s)
    brake: vehicle brake state
    signal: vehicle turn signal state
    lane_turn: lane turn type
    lane_hash: hashed land id
    bool: if bool is False vehicle feature is missing at timestamp
    '''
    def __init__(self, data_path):

        self.data_path = data_path
        self.needed_feature_key = ['TRAJECTORY_FEATURE', 'SL_TRAJECTORY_FEATURE', 'FORWARD_MAP_FEATURE', 'NEIGHBOR_FEATURE']
        self.load_data()

    def load_data(self):
        self.feature_list = [[] for i in range(4)]

        data_dict = np.load(self.data_path, allow_pickle=True).item()
        data_list = list(data_dict.values())
        np.random.shuffle(data_list)
        count = 0
        for value in data_list:
            for i, feature_key in enumerate(self.needed_feature_key):
                curr_feature = value[feature_key]
                curr_feature = np.array(curr_feature)
                self.feature_list[i].append(curr_feature)

    def __len__(self):
        return len(self.feature_list[0])

    def __getitem__(self, idx):
        try:
            now_trajectory = self.feature_list[0][idx]
            now_sl_trajectory = self.feature_list[1][idx]
            now_forward_map_feature = self.feature_list[2][idx]
            now_neighbor_feature = self.feature_list[3][idx]

            return now_sl_trajectory, now_trajectory, now_forward_map_feature, now_neighbor_feature

        except Exception as e:
            print(e)
            new_index = np.random.randint(0, len(self) - 1)
            return self[new_index]


if __name__ == "__main__":
    data_path = "/data/dataset/learn_data_0_30_50_on_lane.npy"
    dataset = TestDataset(data_path)
    print("dataset size: %d"%(len(dataset)))
    now_feature = list(dataset.__getitem__(0))
    print("sample feature shape:")
    for feature_key, feature in zip(dataset.needed_feature_key, now_feature):
        print("%s shape: %s"%(feature_key, feature.shape))
