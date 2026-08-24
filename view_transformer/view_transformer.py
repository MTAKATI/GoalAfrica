import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        # Define vertices with explicit float32 type
        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ], dtype=np.float32)

        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ], dtype=np.float32)

        # Calculate the transform matrix
        self.perspective_transform_matrix = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        
        # Check if point is inside the defined polygon
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None 
        
        # Reshape point for perspectiveTransform (expects 3D array: [N, 1, 2])
        reshape_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshape_point, self.perspective_transform_matrix)

        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    
                    tracks[object_name][frame_num][track_id]['position_transformed'] = position_transformed