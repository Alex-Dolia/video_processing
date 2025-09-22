from utils_split_merge.utils import inverse_cosine_similarity_fn, filter_out_bad_small_faces, face_id_graph_clustering, find_threshold
from utils_split_merge.NEW_utils import NEW_inverse_cosine_similarity_fn, NEW_add_quality_related_columns
from scipy.optimize import linear_sum_assignment as linear_assignment
from collections import OrderedDict
import itertools
import json
import numpy as np
import pandas as pd
import os
INFTY_COST = 1e+5

def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric.copy()  #(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(tracks[track_idx])
            unmatched_detections.append(detections[detection_idx])
        else:
            matches.append({"first": tracks[track_idx], "second": detections[detection_idx], "connected": True})
    return matches, unmatched_tracks, unmatched_detections

def find_maximal_sets_new(frames):
    """
    We find what faces are appeared in the same frame.
    For example, frames = [{1} {1, 2} {1, 2, 3} {1, 3} {3} {3, 4} {4} {5} {5, 6} {6}],
    where {1} and {1, 2, 3} means we have only a face with id 1 and 
    faces with ids 1, 2 and 3 in the corresponding frames.
    
    Then we get: maximal_sets = [{1, 2, 3}, {3, 4}, {5, 6}].
    """
    # Sort frames by length in descending order
    frames = [set(i) for i in OrderedDict.fromkeys(frozenset(item) for item in frames)]
    sorted_frames = sorted(frames, key=len, reverse=True)

    maximal_sets = []
    for i, frame in enumerate(sorted_frames):
        is_maximal = True
        for larger_frame in maximal_sets:
            if frame < larger_frame:
                is_maximal = False
                break
        if is_maximal:
            maximal_sets.append(frame)

    maximal_sets = [set(i) for i in OrderedDict.fromkeys(frozenset(item) for item in maximal_sets)]
    return maximal_sets

def is_subset_of_any(current_set, list_of_sets):
    """
    Check if the current_set is a subset of at least one set in list_of_sets.
    
    Parameters:
    current_set (set): The set to check.
    list_of_sets (list): A list of sets to check against.
    
    Returns:
    bool: True if current_set is a subset of at least one set in list_of_sets, False otherwise.
    """
    for s in list_of_sets:
        if current_set.issubset(s):
            return True
    return False

def NEW_not_in_same_frame_check(quality_movie_tracker_output, new_inverse_cosine_similarity):
    """
    add the column "not_in_same_frame" to new_inverse_cosine_similarity
    """
    selected_cols = ['n', 'track_id']
    n_track_ids = [{key: value for key, value in row.items() if key in selected_cols} for row in quality_movie_tracker_output]
    n_track_ids_df = pd.DataFrame(n_track_ids)


    same_frame_track_ids = n_track_ids_df.copy()
    same_frame_track_ids["#"] = 1
    same_frame_track_ids = same_frame_track_ids.groupby(["n"]).agg({"#": "sum", "track_id": set})
    sub_same_frame_track_ids = same_frame_track_ids[same_frame_track_ids["#"] > 1].copy()

    sub_same_frame_track_ids = sub_same_frame_track_ids["track_id"].values.copy()
    sub_same_frame_track_ids = find_maximal_sets_new(sub_same_frame_track_ids)
    """
    find_maximal_sets_new finds what faces are appeared in the same frame.
    For example, frames = [{1} {1, 2} {1, 2, 3} {1, 3} {3} {3, 4} {4} {5} {5, 6} {6}],
    where {1} and {1, 2, 3} means we have only a face with id 1 and 
    faces with ids 1, 2 and 3 in the corresponding frames.
    
    Then we get: maximal_sets = [{1, 2, 3}, {3, 4}, {5, 6}].
    """    
    new_inverse_cosine_similarity["not_in_same_frame"] = new_inverse_cosine_similarity.apply(lambda x: not ((x["first_track_id"] == x["second_track_id"]) or is_subset_of_any({x["first_track_id"], x["second_track_id"]}, sub_same_frame_track_ids)), axis = 1) 
    return new_inverse_cosine_similarity

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
           request_data["constant_height_theshold"] = 55
        
        if "constant_width_theshold" not in request_data:     
           request_data["constant_width_theshold"] = 55
        
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
    
def select_good_faces(inv_cos_df,
                      height_theshold,
                      width_theshold,
                      face_quality_theshold,
                      good_min_or_max,
                      good_faces_selection_criterion):
    
    # we want to compare faces that have good sizes and face qualities
    good_inv_cos_df = inv_cos_df[(inv_cos_df["worst_height"] >= height_theshold)].copy() 
    good_inv_cos_df = good_inv_cos_df[(inv_cos_df["worst_width" ] >= width_theshold)].copy()
    good_inv_cos_df = good_inv_cos_df[(good_inv_cos_df["worst_face_quality"] >= face_quality_theshold)].copy()
    
    # I think you can filter the output of faces = add_face_quality(faces,pose_detector,show = False)
    # with faces smaller than 55*55 ,
    # FFTBlurriness<1000,  sharpness > 400   
    good_inv_cos_df = good_inv_cos_df[(good_inv_cos_df["worst_sharp_score"] <= 400)].copy()
    good_inv_cos_df = good_inv_cos_df[(good_inv_cos_df["worst_blur_score" ] >= 1000)].copy()

    display(good_inv_cos_df.head())
    good_inv_cos_df = good_inv_cos_df[["first_n", "second_n", "first_track_id", "second_track_id", "worst_face_quality","worst_sharp_score","worst_blur_score",   "inverse_cosine"]].copy()

    if len(good_inv_cos_df) > 0:
       good_inv_cos_df = good_inv_cos_df.loc[good_inv_cos_df.groupby(["first_track_id", "second_track_id"])[good_faces_selection_criterion].transform(good_min_or_max) ==  good_inv_cos_df[good_faces_selection_criterion]].copy()
       good_inv_cos_df.drop_duplicates(subset = ["first_track_id", "second_track_id"], inplace = True)
       good_inv_cos_df["isGood"] = True
    return good_inv_cos_df

def select_bad_faces(inv_cos_df, 
                     good_inv_cos_df, 
                     bad_min_or_max,
                     bad_faces_selection_criterion,
                     useBadFace):
    if not useBadFace:
           bad_inv_cos_df = pd.DataFrame(columns = inv_cos_df.columns)
           return bad_inv_cos_df 
    elif len(good_inv_cos_df) == 0: # NO good faces ALL BAD FACES
         bad_inv_cos_df = inv_cos_df.copy()   
         print("elif len(good_inv_cos_df) == 0: # NO good faces ALL BAD FACES")
         print("len(bad_inv_cos_df): ", len(bad_inv_cos_df))
    else:    
       # there are good faces    
       sub_good_inv_cos_df = good_inv_cos_df[['first_track_id', 'second_track_id']].drop_duplicates()
       # Merge with indicator to find rows that are only in inv_cos_df but not in good_inv_cos_df:
       bad_inv_cos_df = inv_cos_df.merge(sub_good_inv_cos_df, on=['first_track_id', 'second_track_id'], how='left', indicator=True)
       # Filter rows that are only in inv_cos_df
       bad_inv_cos_df = bad_inv_cos_df[bad_inv_cos_df['_merge'] == 'left_only'].drop(columns='_merge')

    print("if len(bad_inv_cos_df) > 0: ", len(bad_inv_cos_df) > 0, len(bad_inv_cos_df))
    if len(bad_inv_cos_df) > 0: 
           bad_inv_cos_df = bad_inv_cos_df.loc[bad_inv_cos_df.groupby(["first_track_id", "second_track_id"])[bad_faces_selection_criterion].transform(bad_min_or_max) ==  bad_inv_cos_df[bad_faces_selection_criterion]].copy()
           bad_inv_cos_df.drop_duplicates(subset = ["first_track_id", "second_track_id"], inplace = True)
           bad_inv_cos_df["isGood"] = False
    return bad_inv_cos_df

def NEW_predict_fn(input_data, 
                   model_trans,
                   useBadFace = True,
                   embedding_type = "FaceNet512",
                   good_faces_selection_criterion = "inverse_cosine",
                   bad_faces_selection_criterion =  "inverse_cosine", #"worst_face_quality",
                   ):

    print("MERGE NEW_print_fn, len(input_data): ", len(input_data))
    movie_name = input_data["movie_name"] 
    split_movie_tracker_output = input_data['movie_tracker_output'].copy()
    similarity_threshold = input_data["similarity_threshold"]

    good_faces_selection_criterion = good_faces_selection_criterion.strip()
    bad_faces_selection_criterion = bad_faces_selection_criterion.strip()

    height_theshold = input_data["constant_height_theshold"]
    width_theshold = input_data["constant_width_theshold"]
    face_quality_theshold = input_data["constant_face_quality_theshold"]
    
    
    print("movie_name: ", movie_name)
    print("similarity_threshold: ", similarity_threshold)
    print("height_theshold: ", height_theshold)
    print("width_theshold: ", width_theshold)
    print("face_quality_theshold: ", face_quality_theshold)
    print("good_faces_selection_criterion: ", good_faces_selection_criterion)
    print("bad_faces_selection_criterion: ", bad_faces_selection_criterion)

    good_min_or_max = "min" if good_faces_selection_criterion == "inverse_cosine" else "max"
    bad_min_or_max  = "min" if bad_faces_selection_criterion  == "inverse_cosine" else "max"
    
    del input_data
    """
    split_movie_tracker_output = [
                                         {"n": 1, "track_id": 1, "embedding": np.array([1, 0])},
                                         {"n": 2, "track_id": 2, "embedding": np.array([1, 1])},
                                         {"n": 2, "track_id": 3, "embedding": np.array([0, 1])}
                                 ]
    
    """
    inv_cos_df = NEW_inverse_cosine_similarity_fn(split_movie_tracker_output, face_embedding_name = 'embedding')
    
    # quality_related_columns: width, height and quality of face.
    inv_cos_df = NEW_add_quality_related_columns(split_movie_tracker_output, inv_cos_df)
    
    # check if two tracks have common frame - that is two different face appears in the same frame
    inv_cos_df = NEW_not_in_same_frame_check(split_movie_tracker_output, inv_cos_df) 
    inv_cos_df = inv_cos_df[inv_cos_df["not_in_same_frame"]].copy()
    
    del split_movie_tracker_output, inv_cos_df["not_in_same_frame"]

    # find the worst quality face between two of them
    # if selection_criterion = "worst_face_quality" we want to find the best worst face pair from the two given tracks
    inv_cos_df["worst_face_quality"] = inv_cos_df.apply(lambda x: min(x["first_face_quality"], x["second_face_quality"]), axis = 1) 
    inv_cos_df["worst_height"] = inv_cos_df.apply(lambda x: min(x["first_height"], x["second_height"]), axis = 1)
    inv_cos_df["worst_width"] = inv_cos_df.apply(lambda x: min(x["first_width"], x["second_width"]), axis = 1)
    
    print("MERGE inv_cos first_sharp_score, second_sharp_score, first_blur_score and second_blur_score")
    print(inv_cos_df[["first_sharp_score", "second_sharp_score", "first_blur_score", "second_blur_score"]].head())
    inv_cos_df["worst_sharp_score"] = inv_cos_df.apply(lambda x: max(x["first_sharp_score"], x["second_sharp_score"]), axis = 1)
    inv_cos_df["worst_blur_score" ] = inv_cos_df.apply(lambda x: min(x["first_blur_score"],  x["second_blur_score"]),  axis = 1)    

    merge_predict_path = f'results/unique_faces/{embedding_type}/inv_cos/{movie_name}'
    os.makedirs(merge_predict_path, exist_ok=True)
    inv_cos_df.to_csv(f"{merge_predict_path}/inv_cos_df_same_frame_check.csv")

    inv_cos_df = inv_cos_df[["first_n", "second_n", "first_track_id", "second_track_id", "worst_height", "worst_width", "worst_face_quality", "worst_sharp_score", "worst_blur_score", "inverse_cosine"]].copy()
    print("------------!!!!!!!!!!!!!!!!!!!!!------------------before select_good_faces: ")
    print(inv_cos_df.head())
    # we want to compare faces that have good sizes and face qualities
    good_inv_cos_df = select_good_faces(inv_cos_df,
                                        height_theshold,
                                        width_theshold,
                                        face_quality_theshold,
                                        good_min_or_max,
                                        good_faces_selection_criterion)

    bad_inv_cos_df = select_bad_faces(inv_cos_df, 
                                      good_inv_cos_df, 
                                      bad_min_or_max,
                                      bad_faces_selection_criterion,
                                      useBadFace)

    selected_inv_cos_df = pd.DataFrame(columns = good_inv_cos_df.columns)
    selected_frames_df = pd.DataFrame(columns = ["first_n", 
                                              "second_n", 
                                              "first_track_id", 
                                              "second_track_id", 
                                              "worst_face_quality",
                                              "worst_sharp_score",
                                              "worst_blur_score",
                                              "inverse_cosine"])
    #START#####################################################################################
    ###########################################################################################
    ###########################################################################################
    if len(good_inv_cos_df) > 0:
           selected_inv_cos_df = pd.concat([selected_inv_cos_df, good_inv_cos_df]) 

           good_selected_frames_df = inv_cos_df.merge(good_inv_cos_df[["first_track_id", "second_track_id", good_faces_selection_criterion]], on = ["first_track_id", "second_track_id", good_faces_selection_criterion])[["first_n", "second_n", "first_track_id", "second_track_id", "worst_face_quality", "worst_sharp_score", "worst_blur_score", "inverse_cosine"]].copy() 
           selected_frames_df = pd.concat([selected_frames_df, good_selected_frames_df])

    if len(bad_inv_cos_df) > 0:
           selected_inv_cos_df = pd.concat([selected_inv_cos_df, bad_inv_cos_df])

           bad_selected_frames_df = inv_cos_df.merge(bad_inv_cos_df[["first_track_id", "second_track_id",  bad_faces_selection_criterion]], on = ["first_track_id", "second_track_id", bad_faces_selection_criterion])[["first_n", "second_n", "first_track_id", "second_track_id", "worst_face_quality", "worst_sharp_score", "worst_blur_score", "inverse_cosine"]].copy() 
           selected_frames_df = pd.concat([selected_frames_df, bad_selected_frames_df])
    # output is the following: 
    # a) selected_inv_cos_df - it is used for clustering;
    # b) selected_frames - it can be used to see faces that were selected for the merging 
    # suffix n and track_id define frame number and track id, 
    # for example "first_n" and "first_track_id" or "second_n" and "second_track_id"
    #END#######################################################################################
    ###########################################################################################
    ###########################################################################################    
    movie_name = movie_name.split(".")[0]
    merge_predict_path = f"results/unique_faces/{embedding_type}/selected_frames_df/{movie_name}/{face_quality_theshold}"
    os.makedirs(merge_predict_path, exist_ok=True)

    selected_frames_df.drop_duplicates(subset = ["first_track_id", "second_track_id"], inplace = True)
    selected_frames_df.to_csv(f"{merge_predict_path}/{movie_name}_goodfaces_{good_faces_selection_criterion}_badfaces_{bad_faces_selection_criterion}_useBadFace_{useBadFace}_selected_frames_df.csv")

    selected_inv_cos_df["first"] = selected_inv_cos_df["first_track_id"].copy()
    selected_inv_cos_df["second"] = selected_inv_cos_df["second_track_id"].copy()
    selected_inv_cos_df = selected_inv_cos_df[["first", "second", "inverse_cosine"]].copy()
    merged_map_track_ids = {}
    print("selected_inv_cos_df:")
    print(selected_inv_cos_df)
    if len(selected_inv_cos_df) > 0:
             output_elem, mapping_old_to_new = face_id_graph_clustering(selected_inv_cos_df, threshold = similarity_threshold)
             clusters = output_elem["clusters"]
             print("!!! clusters: ", clusters)   
             merged_map_track_ids = {int(value): int(cluster[0]) for cluster in clusters for value in sorted(cluster)}

    return merged_map_track_ids

        
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

