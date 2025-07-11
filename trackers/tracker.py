from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
# from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        detections = self.detect_frames(frames)


    def detect_frames(self, frames):
        batch_size=20 # Frames processed in batches for efficiency and speed.
        detections = [] 
        for i in range(0,len(frames),batch_size):
            # Iterating through the list of input frames using batch-sized increments, slicing them into chunks and processing them using the predict method.
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()} # Allow classes to be referred to by name rather than id number.

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)


            # Convert goalkeeper objects to player objects
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Tracker object added to each detection
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist() # bounding box
                cls_id = frame_detection[3] # class id
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            # The ball has been filtered out of the detection_with_tracks. 
            # A second loop is therefore required using the original detection_supervision. 
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist() # bounding box
                cls_id = frame_detection[3] # class id

                if cls_id == cls_names_inv['ball']:
                    track_id = 1  # The ball is not tracked across frames, so we assign a fixed tracker ID of 1. 
                    tracks["ball"][frame_num][track_id] = {"bbox":bbox} # This works because only one ball detection is expected per frame.
                    
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks