import sys
import numpy as np
import cv2
from utils import get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self, frame_rate=24, window_size=5):
        self.frame_rate = frame_rate
        self.frame_windows = window_size

    def calculate_speed_and_distance(self, tracks):
        for obj_type, object_tracks in tracks.items():
            if obj_type in ['ball', 'referees', 'referee']:
                continue

            total_distances = {}

            for frame_idx in range(0, len(object_tracks) - self.frame_windows, self.frame_windows):
                future_idx = frame_idx + self.frame_windows

                for track_id, track in object_tracks[frame_idx].items():
                    if track_id not in object_tracks[future_idx]:
                        continue

                    p1 = track.get('position_transformed')
                    p2 = object_tracks[future_idx][track_id].get('position_transformed')

                    if p1 is None or p2 is None:
                        continue

                    # Euclidean distance in meters
                    distance_meters = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    time_seconds = self.frame_windows / self.frame_rate
                    
                    speed_m_s = distance_meters / time_seconds
                    speed_kmh = speed_m_s * 3.6

                    # CALIBRATION: Cap realistic human sprinting speed (~36 km/h max)
                    if speed_kmh > 36.0:
                        continue

                    # Accumulate distance
                    total_distances[track_id] = total_distances.get(track_id, 0) + distance_meters

                    # Apply smoothed speed and distance to frame range
                    for f in range(frame_idx, future_idx):
                        if track_id in object_tracks[f]:
                            object_tracks[f][track_id]['speed'] = speed_kmh
                            object_tracks[f][track_id]['distance'] = total_distances[track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == 'ball' or object == 'referees' or object == 'referee':
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:
                            continue

                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(frame)

        return output_frames