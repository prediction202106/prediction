import argparse
import glob
import os
from typing import Union, List, Dict
from scipy.spatial import KDTree
import numpy as np
from pathlib import Path

from shapely.geometry import LineString
from shapely.geometry.point import Point

import google.protobuf.text_format as text_format

from proto.hdmap_pb2 import HdMap, Lane

class LaneSegment():
    def __init__(self, lane: Lane):
        self.lane = lane
        self.segment = []
        self.heading = []
        self.accmulated_s = []
        self.build_segment()
        self._tree = KDTree(self.segment)

    @property
    def tree(self) -> KDTree:
        return self._tree

    def query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf, workers=1):
        self._tree.query(x, k, eps, p, distance_upper_bound, workers)

    def build_segment(self):
        self.lane_id = self.lane.id.id
        self.lane_type = self.lane.type
        self.lane_turn = self.lane.turn
        if self.lane.HasField("central"):
            points = self.lane.central.point
            if len(points) <= 0:
                return
            prev_p = points[0].point
            self.segment.append([prev_p.x, prev_p.y])
            self.accmulated_s.append(0)
            for i in range(1, len(points)):
                p = points[i].point
                dx = p.x - prev_p.x
                dy = p.y - prev_p.y
                theta = np.arctan2(dy, dx)
                s = np.hypot(dy, dx) + self.accmulated_s[-1]
                self.segment.append([p.x, p.y])
                self.heading.append(theta)
                self.accmulated_s.append(s)
                prev_p = p
        self.accmulated_s = np.array(self.accmulated_s, dtype="float")

    def get_smooth_point(self, s) -> Point:
        lane_string = LineString(self.segment)
        s = max(min(s, lane_string.length), 0)
        point = lane_string.interpolate(s)
        return point

    def get_heading(self, s):
        if s <= 0:
            return self.heading[0]
        if s >= self.accmulated_s[-1]:
            return self.heading[-1]
        idx = np.sum(self.accmulated_s < s) - 1
        return self.heading[idx]

    def get_point_from_lane(self, s, l):
        smooth_point = self.get_smooth_point(s)
        heading = self.get_heading(s)
        return Point(smooth_point.x - l * np.sin(heading), smooth_point.y + l * np.cos(heading))

    @property
    def length(self):
        return self.lane.length

    @property
    def id(self):
        return self.lane_id

    @property
    def successor_id(self):
        return self.lane.successor_id

    @property
    def predecessor_id(self):
        return self.lane.predecessor_id

    @property
    def left_neighbor_forward_lane_id(self):
        return self.lane.left_neighbor_forward_lane_id

    @property
    def right_neighbor_forward_lane_id(self):
        return self.lane.right_neighbor_forward_lane_id
    


class HDMapUtil:
    def __init__(self, map_file=""):
        self.map_file = Path(map_file)
        self.hdmap = HdMap()
        self.lane_hash_dict = {}
        self.lane_segment_dict = {}
        if self.map_file:
            self.setup(self.map_file)
        self.build_lane_dict()

    def setup(self, map_file: Path):
        self.get_pb_from_file(map_file, self.hdmap)
        hash_file = map_file.parent.joinpath("lane_hash.txt")
        with open(hash_file) as f:
            for line in f.readlines():
                line = line.strip().split(":")
                lane_id = line[0].strip()
                lane_hash = line[1].strip()
                # print("%s: %s"%(lane_hash, lane_id))
                self.lane_hash_dict[lane_hash] = lane_id
        print("hdmap lane num: %d" % len(self.lane_hash_dict.keys()))

    def get_pb_from_text_file(self, filename, pb_value):
        """Get a proto from given text file."""
        with open(filename, 'r') as file_in:
            return text_format.Merge(file_in.read(), pb_value)

    def get_pb_from_bin_file(self, filename, pb_value):
        """Get a proto from given binary file."""
        with open(filename, 'rb') as file_in:
            pb_value.ParseFromString(file_in.read())
        return pb_value

    def get_pb_from_file(self, filename, pb_value):
        """Get a proto from given file by trying binary mode and text mode."""
        try:
            return self.get_pb_from_bin_file(filename, pb_value)
        except:
            try:
                return self.get_pb_from_text_file(filename, pb_value)
            except:
                print('Error: Cannot parse %s as binary or text proto' % filename)
        return None

    def build_lane_dict(self):
        for lane in self.hdmap.lane:
            lane_id = lane.id.id
            lane_segment = LaneSegment(lane)
            if len(lane_segment.segment) < 2:
                print(lane_id)
            self.lane_segment_dict[lane_id] = lane_segment
        return

    def get_lane_id_by_hash(self, lane_hash) -> str:
        if isinstance(lane_hash, str):
            str_lane_hash = lane_hash
        elif isinstance(lane_hash, int):
            str_lane_hash = str(lane_hash)
        else:
            str_lane_hash = str(int(lane_hash))
        try:
            lane_id = self.lane_hash_dict[str_lane_hash]
        except:
            lane_id = ""
        return lane_id

    def get_lane_by_id(self, lane_id) -> LaneSegment:
        lane_segment = self.lane_segment_dict.get(lane_id, None)
        return lane_segment

    def get_lanes(self, center_point, search_radius) -> List[int]:
        lane_list = []
        for lane_id, lane_segment in self.lane_segment_dict.items():
            distance, idx = lane_segment.tree.query(center_point)
            if distance < search_radius:
                lane_list.append([lane_id, distance, idx])
        lane_list = [l for l in sorted(lane_list, key=lambda x: x[1])]
        return lane_list

if __name__ == "__main__":
    root_path = os.path.abspath("./")
    map_file = os.path.join(root_path, "/data/map/base_map.bin")
    assert(os.path.isfile(map_file))
    hdmap = HDMapUtil(map_file)
    lane_hash = "1411311313"
    lane_id = hdmap.get_lane_id_by_hash(lane_hash)
    assert(lane_id)
    print("current lane: %s: "%lane_id)
    lane = hdmap.get_lane_by_id(lane_id)
    assert(lane)
    
    print("current lane length: %f"%lane.length)
    print("lane successor_ids: ", lane.successor_id)
    print("lane predecessor_id: ", lane.predecessor_id)
    print("lane left_neighbor_forward_lane_id: ", lane.left_neighbor_forward_lane_id)
    print("lane right_neighbor_forward_lane_id: ", lane.right_neighbor_forward_lane_id)
    
    query_s = 1.0
    query_l = -0.5
    smooth_point = lane.get_smooth_point(query_s)
    print("point at [s = %f, l = 0.0] of the lane: (%f, %f)"%(query_s, smooth_point.x, smooth_point.y))
    point = lane.get_point_from_lane(1.0, 0.5)
    print("point at [s = %f, l = %f] of the lane: (%f, %f)"%(query_s, query_l, point.x, point.y))
    heading = lane.get_heading(1.0)
    print("heading at [s = %f] of the lane: %f(degree)"%(query_s, heading / 3.14 * 180))

    lane_list = hdmap.get_lanes(point, 5)
    print("nearby lanes at point (%f, %f): %s"%(point.x, point.y, lane_list))