import torch, os, json, io, cv2, time, numpy as np
from ultralytics import YOLO
import datetime 
import boto3

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    model_name =  'yolov8n-face.pt'
    model = YOLO(f"model/{model_name}")
    return model

def input_fn(request_body, request_content_type): 
    def  write_read_s3_movie(request_body):
        
         s3_bucket, s3_key, image_file_name, local_dir = request_body.split("|")
         s3_key = f"{s3_key}/{image_file_name}"
        
         # Ensure the local directory exists
         if not os.path.exists(local_dir):
                os.makedirs(local_dir)
        
         # Example of writing to the local file system
         local_file_path = os.path.join(local_dir, image_file_name)  
        
         try:  
             if not (s3_bucket == "LOCAL"): 
                     # Download the file from S3
                     s3 = boto3.client('s3')
                     print(f"{s3_bucket}/{s3_key}")
                     s3.download_file(s3_bucket, s3_key, local_file_path)
         except Exception as e:
                return local_file_path, "Error Loading Image from s3: " + str(e)
        
         is_path_exist = str(os.path.exists(local_dir))
         output = f"!!! The directory {local_dir}) does exist: {is_path_exist}"
                             
         is_file_exist = str(os.path.exists(local_file_path))   
         ss = f"| The file {local_file_path} does exist: {is_file_exist}"
         output += ss                                                    
         return local_file_path, output
        
    def load_frames_movies(local_file_path):
        frames = []
        try:
          cap = cv2.VideoCapture(local_file_path)

          # List to store frames
          frames = []

          # Read and store frames
          while True:
                ret, frame = cap.read()
                if not ret:
                   break
                # Convert frame to RGB (OpenCV uses BGR by default)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                

          #os.remove(local_file_path)
          # Release the video capture object
          cap.release()
        
          number_of_frames = len(frames)
          frame_movie_loading_status = f"All frames are loaded COMPLETELY: number_of_frames = {number_of_frames}"
        except Exception as e:
               frame_movie_loading_status = "Converting movie to List of Frames is FAILED: " + str(e)
        return frames, frame_movie_loading_status 

    def yield_load_frames_movies(local_file_path):
        frames = []
        cap = cv2.VideoCapture(local_file_path)
        #  ount the number of frames 
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
        fps = cap.get(cv2.CAP_PROP_FPS) 
  
        # calculate duration of the video 
        seconds = round(frames / fps) 
        video_time = datetime.timedelta(seconds=seconds) 
        print("number of frames: ", frames)
        print("movie fps: ", fps)
        print(f"duration in seconds: {seconds}") 
        print(f"video time: {video_time}") 
        
        # Read and store frames
        number_frames = 0
        while True:
              ret, frame = cap.read()
              if not ret:
                 print("Number of Frames: ", number_frames) 
                 break
              # Convert frame to RGB (OpenCV uses BGR by default)
              #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              yield frame
              number_frames += 1

        ####!!!!!!!!! os.remove(local_file_path)
        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()
        
    
    assert request_content_type=='text/csv'
    
    
    local_file_path, status = write_read_s3_movie(request_body)
    print("status: ", status)
    #frames, frame_movie_loading_status = load_frames_movies(local_file_path)
    frames=[1,2]
    frame_movie_loading_status = ""
    
    status = status + ", load_frames_movies: " + frame_movie_loading_status
    #status = "OK"
    frames = yield_load_frames_movies(local_file_path)
    return {"frames": frames, "status": status}
    
def predict_fn(input_data, model):
    ###########################################################
    def get_track_info(tracks):
        track_ids = None
        boxes = None
        class_ids = None
        confidences = None
        # Get the boxes and track IDs
        if tracks is not None: 
           try:
               boxes = tracks[0].boxes.xyxy.cpu().tolist()
               track_ids = tracks[0].boxes.id.int().cpu().tolist()
               #class_ids = tracks[0].cls.int().cpu().tolist()
               confidences = tracks[0].boxes.conf.cpu().tolist()
               class_ids =[]
               for track in tracks:
                   class_ids.append(track.cls) 
           except:
               e = True  
        return track_ids, boxes, confidences, class_ids 

    def tracking(model,
                 frames, 
                 persist = True, 
                 show = False, 
                 classes_to_count = [0], # face 
                 tracker = "botsort.yaml"):

        movie_tracker_output = []
        nufs = []
        
        try:
          nframes = 0
          for i, frame in enumerate(frames):
              if i % 100 == 0:
                 print(f"Tracking frame {i}")
              nframes += 1
              if frame is not None:
                 conf = 0.3
                 iou = 0.3
                 with torch.no_grad():
                      tracks = model.track(frame,
                                           persist=True, 
                                           show=False, 
                                           conf=conf, 
                                           iou=iou, 
                                           classes=classes_to_count, 
                                           tracker=tracker,
                                           verbose=False)

                 track_ids, boxes, confidences, class_ids = get_track_info(tracks)
                 
                 if track_ids is not None:
                    for track_id, bounding_box, confidence in zip(track_ids, boxes, confidences):
                        
                        if (confidence > 0.0):
                            x1, y1, x2, y2 = bounding_box 
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
           
                            face = frame[y1:y2, x1:x2]
                            height, width, channels = face.shape
                        
                            elem = {"n": int(i),
                                    "frame": frame,  
                                    "face": face,
                                    "height": height, 
                                    "width": width, 
                                    "channels": channels,
                                    "track_id": int(track_id), 
                                    "bounding_box": bounding_box, 
                                    "confidence": confidence}

                            movie_tracker_output.append(elem)
                            nufs.append(track_id)
          
          number_of_tracks = len(set(nufs))
           
          status = f"Tracking is finished COMPLETELY! Number of frames: {nframes} | There are {number_of_tracks} tracks!"    
        except Exception as e:
               status = "Tracking FAILED: " + str(e) + f", processed {nframes} frames"
        
        nufs = len(set(nufs))
        print("!!!!!!!!! Number of track ids after tracking: ", nufs)
        return movie_tracker_output, nufs, status

    ###########################################################
    status = input_data["status"]
    frames = input_data["frames"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    movie_tracker_output, nufs, tracking_status = tracking(model, frames)
    status += "| Tracking Status: " + tracking_status
    
    del model   
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    return {"movie_tracker_output": movie_tracker_output, "nufs": nufs, "status": status} 

    
def output_fn(prediction_output, content_type):
    class NumpyEncoder(json.JSONEncoder):
          def default(self, obj):
              if isinstance(obj, np.ndarray):
                 return obj.tolist()
              return json.JSONEncoder.default(self, obj)

    return json.dumps(prediction_output, cls = NumpyEncoder)