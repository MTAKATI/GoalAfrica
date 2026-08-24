import numpy as np
import cv2

class ViewTransformer:
    def __init__(self):
        # Standard pitch dimensions in meters (68m wide x 105m long)
        self.court_width = 68
        self.court_length = 105

        # Source coordinates on frame zero matching the 4 corners of the visible pitch field
        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ], dtype=np.float32)

        # Destination coordinates in meters on 2D tactical map
        self.target_vertices = np.array([
            [0, self.court_width],
            [0, 0],
            [self.court_length, 0],
            [self.court_length, self.court_width]
        ], dtype=np.float32)

        self.perspective_transform_matrix = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        if cv2.pointPolygonTest(self.pixel_vertices, p, False) < 0:
            return None 
        
        reshape_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshape_point, self.perspective_transform_matrix)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    # Use camera-adjusted position if available
                    position = track_info.get('position_adjusted', track_info['position'])
                    position_transformed = self.transform_point(position)
                    
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    
                    tracks[object_name][frame_num][track_id]['position_transformed'] = position_transformed

    def draw_tactical_map(self, frame, tracks, frame_num, map_w=300, map_h=200):
        # Create a black background box in the top-left corner
        map_bg = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        cv2.rectangle(map_bg, (0, 0), (map_w, map_h), (34, 139, 34), -1)
        cv2.rectangle(map_bg, (0, 0), (map_w - 1, map_h - 1), (255, 255, 255), 2)
        cv2.line(map_bg, (map_w // 2, 0), (map_w // 2, map_h), (255, 255, 255), 1)

        # Draw players on top-down mini-map
        players = tracks['players'][frame_num]
        for player_id, player_info in players.items():
            pos_2d = player_info.get('position_transformed')
            if pos_2d is None:
                continue

            # Scale meters (105x68) to map pixels (300x200)
            map_x = int((pos_2d[0] / self.court_length) * map_w)
            map_y = int((pos_2d[1] / self.court_width) * map_h)
            
            color = player_info.get('team_color', (0, 0, 255))
            cv2.circle(map_bg, (map_x, map_y), 5, color, -1)

        # Draw ball on top-down mini-map
        ball = tracks['ball'][frame_num].get(1, {})
        ball_pos_2d = ball.get('position_transformed')
        if ball_pos_2d is not None:
            map_x = int((ball_pos_2d[0] / self.court_length) * map_w)
            map_y = int((ball_pos_2d[1] / self.court_width) * map_h)
            cv2.circle(map_bg, (map_x, map_y), 4, (0, 255, 255), -1)

        # Overlay map onto original frame (top-left alignment)
        frame[20:20 + map_h, 20:20 + map_w] = map_bg
        return frame