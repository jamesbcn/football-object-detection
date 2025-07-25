import numpy as np 
import cv2

class ViewTransformer():
    def __init__(self):
        pitch_width = 68
        pitch_length = 23.32

        self.pixel_vertices = np.array([[110, 1035], # Bottom left corner
                               [265, 275], # Top left corner
                               [910, 260], # Top right corner
                               [1640, 915]]) # Bottom right corner
        
        self.target_vertices = np.array([
            [0,pitch_width],
            [0, 0],
            [pitch_length, 0],
            [pitch_length, pitch_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self,point):
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.perspective_transformer)
        return tranform_point.reshape(-1,2)

    def add_transformed_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist() # Squeeze to remove the extra dimension from the NumPy array, tolist to convert to a python list
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed