import numpy as np
import pandas as pd
import json
from utils_split_merge.utils import inverse_cosine_similarity_fn, filter_out_bad_small_faces, face_id_graph_clustering, find_threshold

def split_track_id(unpivot_cosine_similarities_df,
                   all_faces_ids_df, 
                   good_quality_and_size_faces_ids_df,
                   threshold):

    track_ids = all_faces_ids_df["track_id"].unique()
    max_track_id = max(track_ids)
          
    new_all_faces_ids_df = pd.DataFrame()
    # Look at one track_id at a time trying to figure out 
    # if we have more than 1 different faces in the given track_id
    for track_id in track_ids:     
        inv_cosine = unpivot_cosine_similarities_df[unpivot_cosine_similarities_df["first_track_id"] == track_id].copy() 
        inv_cosine = inv_cosine[inv_cosine["first_track_id"] == inv_cosine["second_track_id"]].copy()
        # therefore, inv_cosine contain inv cosine distance between faces that belongs to track_id only 

        all_df = all_faces_ids_df[all_faces_ids_df["track_id"] == track_id].copy()
        all_df.sort_values(by = "n", ascending = True, inplace = True)
   
        df = good_quality_and_size_faces_ids_df[good_quality_and_size_faces_ids_df["track_id"] == track_id].copy()
        # if we have only one good face per track or no good faces at all we can do nothing
        # and assume the track has the only one unique face
        if len(df) <= 1:
           all_df["new_track_id"] = all_df["track_id"].copy()
        else:    
           good_n_frames = df["n"].values # good frames for the given track_id 
           # we have more than 1 good face in the track and need to check
           # if we can split it or find faces belonning to the different people  

           # we select frames that contain only good quility faces
           good_inv_cosine = inv_cosine[inv_cosine["first_n"].isin(good_n_frames) & inv_cosine["second_n"].isin(good_n_frames)].copy()
           
           output_elem, _ = face_id_graph_clustering(good_inv_cosine, threshold)

           # clusters is a list where every element is the list containing frame ids of the same faces
           # if clusters has the only one element then it mean we have only one face in the track 
           clusters = output_elem["clusters"].copy()
           clusters = sorted(clusters)

           if len(clusters) > 1:   
              new_track_ids = [track_id]
              for i in range(1, len(clusters)):
                  max_track_id += 1
                  new_track_ids.append(max_track_id)

              # for every frame id assign new track id if track_id changed or keep the old one
              mapping_old_new_track_ids = [{"n": int(clusters[i][k].split("_")[0]), "new_track_id": new_track_ids[i]} for i, cluster in enumerate(clusters) for k in range(len(clusters[i]))]   
              mapping_old_new_track_ids = pd.DataFrame(mapping_old_new_track_ids)

              # recall all_df contains frames that belong to the only one originak track_id 
              all_df = all_df.merge(mapping_old_new_track_ids, on = "n", how = "left")  
              # after good face in the track we assume all faces with bad quality have the same track_id
              # until the next good face
              all_df["new_track_id"] = all_df["new_track_id"].ffill()
              # if the first good face is not the first frame of the track
              # then from the first frame until the first good face we assume
              # the track id is the same as the first good face
              all_df["new_track_id"] = all_df["new_track_id"].bfill()   
           else: 
              all_df["new_track_id"] = all_df["track_id"].copy()
        
        new_all_faces_ids_df = pd.concat([new_all_faces_ids_df, all_df])

    split_map_track_ids = {str(int(n)) + "_" + str(int(track_id)): int(new_track_id)  for n, track_id, new_track_id in new_all_faces_ids_df[["n", "track_id", "new_track_id"]].values if track_id != new_track_id}
    return split_map_track_ids

def model_fn(model_dir):
    return None

def input_fn(request_body, request_content_type):
    """
    Preprocess the incoming request.

    Parameters:
    request_body (str): The body of the incoming request.
    request_content_type (str): The content type of the incoming request.

    Returns:
    np.array: The preprocessed image.
    """
    # Check the content type
    if request_content_type == 'application/json':
        # Parse the JSON object
        request_data = json.loads(request_body)

        if "constant_height_theshold" not in request_data:   
           request_data["constant_height_theshold"] = 50
        
        if "constant_width_theshold" not in request_data:     
           request_data["constant_width_theshold"] = 50
        
        if "constant_face_quality_theshold" not in request_data:     
           request_data["constant_face_quality_theshold"] = 0.2
        
        if "similarity_threshold" not in request_data:
           # find the threshold for the inv cosine comparison
           # if the inv cosine is below the threshold than two faces are similar
           # not two faces are different otherwise. 
           model_name = "Facenet512"
           request_data["similarity_threshold"] = find_threshold(model_name = model_name, distance_metric = "cosine")
        return request_data

    raise ValueError(f"Unsupported content type: {request_content_type}")
    
def predict_fn(input_data, model_trans):
    quality_movie_tracker_output = input_data['movie_tracker_output'].copy()
    similarity_threshold = input_data["similarity_threshold"]

    constant_height_theshold = input_data["constant_height_theshold"]
    constant_width_theshold = input_data["constant_width_theshold"]
    constant_face_quality_theshold = input_data["constant_face_quality_theshold"]
    
    inverse_cosine_similarity = inverse_cosine_similarity_fn(quality_movie_tracker_output, face_embedding_name = 'embedding')

    all_faces_ids_df, good_quality_and_size_faces_ids_df = \
                      filter_out_bad_small_faces(quality_movie_tracker_output,
                                                 height_theshold = constant_height_theshold,
                                                 width_theshold  = constant_width_theshold,
                                                 face_quality_theshold = constant_face_quality_theshold)

    split_map_track_ids = split_track_id(inverse_cosine_similarity,
                                         all_faces_ids_df, 
                                         good_quality_and_size_faces_ids_df,
                                         similarity_threshold)

    return split_map_track_ids
        
def output_fn(prediction_output, content_type):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return super(NumpyEncoder, self).default(obj)
            
    processed_input_data = prediction_output.copy()
    return json.dumps(processed_input_data, cls = NumpyEncoder)

