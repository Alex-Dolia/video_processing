import numpy as np
import pandas as pd

def csm(A,B):
    """
    A = np.array([[1, 0], [1, 1], [0, 1]])
    B = A
    csm(A, B)
    array([[1.        , 0.70710678, 0.        ],
           [0.70710678, 1.        , 0.70710678],
           [0.        , 0.70710678, 1.        ]])
    """
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)

def inverse_cosine_similarity_fn(emb_quality_movie_tracker_output, face_embedding_name = 'embedding'):
    """
    emb_quality_movie_tracker_output = [
                                         {"n": 1, "track_id": 1, "embedding": np.array([1, 0])},
                                         {"n": 2, "track_id": 2, "embedding": np.array([1, 1])},
                                         {"n": 2, "track_id": 3, "embedding": np.array([0, 1])}
                                       ]
    
    """
    
    
    face_embeddings = [elem[face_embedding_name] for elem in emb_quality_movie_tracker_output]
    n_track_ids = [str(elem["n"]) + "_" + str(elem["track_id"]) 
                   for elem in emb_quality_movie_tracker_output]
   
    face_embeddings = np.array(face_embeddings)
          
    cosine_similarities = csm(face_embeddings, face_embeddings)

    cosine_similarities_df = pd.DataFrame(cosine_similarities, columns = n_track_ids)
    """
    display(cosine_similarities_df)
    	1_1     	2_2      	2_3
    0	1.000000	0.707107	0.000000
    1	0.707107	1.000000	0.707107
    2	0.000000	0.707107	1.000000

    """
    

    cosine_similarities_df["first"] = n_track_ids
    unpivot_cosine_similarities_df = pd.melt(cosine_similarities_df, id_vars="first", var_name = "second", value_vars=n_track_ids)
    """
    display(unpivot_cosine_similarities_df)
    	first	second	value
    0	1_1  	1_1 	1.000000
    1	2_2 	1_1 	0.707107
    2	2_3 	1_1 	0.000000
    3	1_1 	2_2 	0.707107
    4	2_2 	2_2 	1.000000
    5	2_3 	2_2 	0.707107
    6	1_1 	2_3 	0.000000
    7	2_2 	2_3 	0.707107
    8	2_3 	2_3 	1.000000

    """
    
    unpivot_cosine_similarities_df["first_n"] = unpivot_cosine_similarities_df["first"].apply(lambda x: int(x.split("_")[0]))
    unpivot_cosine_similarities_df["first_track_id"] = unpivot_cosine_similarities_df["first"].apply(lambda x: int(x.split("_")[1]))

    unpivot_cosine_similarities_df["second_n"] = unpivot_cosine_similarities_df["second"].apply(lambda x: int(x.split("_")[0]))
    unpivot_cosine_similarities_df["second_track_id"] = unpivot_cosine_similarities_df["second"].apply(lambda x: int(x.split("_")[1]))
 
    unpivot_cosine_similarities_df["inverse_cosine"] = 1 - unpivot_cosine_similarities_df["value"]

    unpivot_cosine_similarities_df = unpivot_cosine_similarities_df[["first", "second", "first_n", "first_track_id", "second_n", "second_track_id", "inverse_cosine"]].copy()
    
    """
    display(unpivot_cosine_similarities_df)
        first	second	first_n	first_track_id	second_n	second_track_id	inverse_cosine
    0	1_1  	1_1 	1   	1           	1        	1            	0.000000e+00
    1	2_2 	1_1 	2   	2           	1       	1           	2.928932e-01
    2	2_3 	1_1 	2   	3           	1       	1           	1.000000e+00
    3	1_1 	2_2 	1   	1           	2        	2           	2.928932e-01
    4	2_2 	2_2 	2   	2            	2        	2           	2.220446e-16
    5	2_3 	2_2 	2   	3            	2        	2           	2.928932e-01
    6	1_1 	2_3 	1   	1           	2        	3           	1.000000e+00
    7	2_2 	2_3 	2   	2           	2       	3           	2.928932e-01
    8	2_3 	2_3 	2   	3           	2       	3           	0.000000e+00

    """
    return unpivot_cosine_similarities_df



def NEW_inverse_cosine_similarity_fn(emb_quality_movie_tracker_output, face_embedding_name = 'embedding'):
    """
    emb_quality_movie_tracker_output = [
                                         {"n": 1, "track_id": 1, "embedding": np.array([1, 0])},
                                         {"n": 2, "track_id": 2, "embedding": np.array([1, 1])},
                                         {"n": 2, "track_id": 3, "embedding": np.array([0, 1])}
                                       ]
    
    """
    face_embeddings = [elem[face_embedding_name] for elem in emb_quality_movie_tracker_output]
    n_track_ids = [str(elem["n"]) + "_" + str(elem["track_id"]) 
                   for elem in emb_quality_movie_tracker_output]
   
    face_embeddings = np.array(face_embeddings)
          
    cosine_similarities = csm(face_embeddings, face_embeddings)

    cosine_similarities_df = pd.DataFrame(cosine_similarities, columns = n_track_ids)
    cosine_similarities_df.index = n_track_ids
    
    
    # Step 2: Extract upper triangular indices excluding the main diagonal
    upper_tri_indices = np.triu_indices_from(cosine_similarities_df, k=1)  # k=1 excludes the diagonal
    # Step 3: Use these indices to create a DataFrame in "long" format
    row_indices, col_indices = upper_tri_indices
    unpivot_cosine_similarities_df = pd.DataFrame({'first': cosine_similarities_df.index[row_indices],
                                                   'second': cosine_similarities_df.columns[col_indices],
                                                   'cosine': cosine_similarities_df.values[row_indices, col_indices]
                                                   })
    
    unpivot_cosine_similarities_df["first_n"] = unpivot_cosine_similarities_df["first"].apply(lambda x: int(x.split("_")[0]))
    unpivot_cosine_similarities_df["first_track_id"] = unpivot_cosine_similarities_df["first"].apply(lambda x: int(x.split("_")[1]))

    unpivot_cosine_similarities_df["second_n"] = unpivot_cosine_similarities_df["second"].apply(lambda x: int(x.split("_")[0]))
    unpivot_cosine_similarities_df["second_track_id"] = unpivot_cosine_similarities_df["second"].apply(lambda x: int(x.split("_")[1]))
 
    unpivot_cosine_similarities_df["inverse_cosine"] = 1 - unpivot_cosine_similarities_df["cosine"]

    cols = ["first", "second", "first_n", "first_track_id", "second_n", "second_track_id", "inverse_cosine"]
    unpivot_cosine_similarities_df = unpivot_cosine_similarities_df[cols].copy()                                                                           
    """
    display(unpivot_cosine_similarities_df)
        first  second   first_n   first_track_id   second_n   second_track_id   inverse_cosine
    0   1_1    2_2      1         1                2          2                 0.292893
    1   1_1    2_3      1         1                2          3                 1.000000
    2   2_2    2_3      2         2                2          3                 0.292893
    """
    return unpivot_cosine_similarities_df

def NEW_add_quality_related_columns(quality_movie_tracker_output, inv_cosine_similarities_df):
    selected_cols = ['n', 'track_id', 'height', 'width', 'confidence', 'face_quality', "sharp_score", "blur_score"]
    
    print("quality_movie_tracker_output[0].keys():")
    print(quality_movie_tracker_output[0].keys())
    print("'track_id' in quality_movie_tracker_output[0].keys()")
    print('track_id' in quality_movie_tracker_output[0].keys())
    
    face_quality_features = [{key: value for key, value in row.items() if key in selected_cols} for row in quality_movie_tracker_output]
    all_faces_ids_df = pd.DataFrame(face_quality_features)
    
    print("all_faces_ids_df:")
    display(all_faces_ids_df.head())
    
    inv_cos_df = inv_cosine_similarities_df.merge(all_faces_ids_df, 
        left_on = ["first_n", "first_track_id"],
        right_on = ["n", "track_id"])

    inv_cos_df.rename(columns = {"height": "first_height", 
                                 "width": "first_width", 
                                 "confidence": "first_confidence",
                                 "face_quality": "first_face_quality",
                                 "sharp_score": "first_sharp_score",
                                 "blur_score": "first_blur_score"
                                }, inplace = True)
    
    print("!!!AFTER rename in NEW_utils: ", inv_cos_df.columns, flush=True)
    
    cols = ["first_n", 
            "first_track_id", 
            "first_height", 
            "first_width", 
            "first_confidence",
            "first_face_quality", 
            "first_sharp_score",
            "first_blur_score",
            "second_n", 
            "second_track_id", 
            "inverse_cosine"]

    print("!!! cols: ", cols, flush=True) 
    print(inv_cos_df.head(), flush=True)
    
    
    inv_cos_df = inv_cos_df[cols].copy()

    inv_cos_df = inv_cos_df.merge(all_faces_ids_df,
        left_on = ["second_n", "second_track_id"],    
        right_on = ["n", "track_id"])


    inv_cos_df.rename(columns = {"height": "second_height", 
                                 "width": "second_width", 
                                 "confidence": "second_confidence",
                                 "face_quality": "second_face_quality",
                                 "sharp_score": "second_sharp_score",
                                 "blur_score": "second_blur_score"
                                }, inplace = True)
    
    cols = cols[:-1] + ["second_height", 
                        "second_width", 
                        "second_confidence", 
                        "second_face_quality", 
                        "second_sharp_score",
                        "second_blur_score",
                        "inverse_cosine"] 
    inv_cos_df = inv_cos_df[cols].copy()
    return inv_cos_df



def filter_out_bad_small_faces(quality_movie_tracker_output,
                               height_theshold = 50,
                               width_theshold  = 50,
                               face_quality_theshold = 0.2):

    selected_cols = ['n', 'track_id', 'height', 'width', 'confidence', 'face_quality']
    face_quality_features = [{key: value for key, value in row.items() if key in selected_cols} for row in quality_movie_tracker_output]
    all_faces_ids_df = pd.DataFrame(face_quality_features) # len = 4992

    good_size_faces_ids_df = all_faces_ids_df[(all_faces_ids_df["height"] >= height_theshold) & (all_faces_ids_df["width" ] >= width_theshold)].copy()
    good_quality_and_size_faces_ids_df = good_size_faces_ids_df[(good_size_faces_ids_df["face_quality"] >= face_quality_theshold)].copy()
             
    # good_quality_and_size_faces_ids_df
    # n  track_id   height   width   confidence   face_quality:
    # 0         1      310     225     0.733119       0.578306

    return all_faces_ids_df, good_quality_and_size_faces_ids_df

class Graph:
 
    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]
 
    def DFSUtil(self, temp, v, visited):
 
        # Mark the current vertex as visited
        visited[v] = True
 
        # Store the vertex to list
        temp.append(v)
 
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
 
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp
 
    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)
 
    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc

def face_id_graph_clustering(results_df, threshold):
    """
    Find good faces that are connected or in the same cluster.
    The number of obtained clusters defines the number of different people.
    """
    if "connected" not in results_df.columns: 
       results_df["connected"] = results_df["inverse_cosine"] <= threshold

    if "not_in_same_frame" in results_df.columns:
       results_df = results_df[results_df["not_in_same_frame"]].copy()

    sub_results_df = results_df[results_df["connected"]].copy()

    # all_nodes contain the connected nodes and possibly single (not connected) nodes
    all_nodes = list(set(list(results_df["first"].values) + list(results_df["second"].values)))
    # nodes contains only connected nodes
    nodes = list(set(list(sub_results_df["first"].values) + list(sub_results_df["second"].values)))
    # we want node to be integer without any gaps
    mapping_old_to_new = {node: i    for i, node in enumerate(nodes)}
    mapping_new_to_old = {i: node for i, node in enumerate(nodes)}

    not_connected_nodes = list(set(all_nodes) - set(nodes))
    not_connected_nodes = [[node] for node in not_connected_nodes]
    # https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
    g = Graph(len(nodes))

    for i, row in sub_results_df.iterrows():
           g.addEdge(mapping_old_to_new[row["first"]], mapping_old_to_new[row["second"]])
    cc = g.connectedComponents()

    result = []
    for i in range(len(cc)):
        one_face= []
        for element in cc[i]:
            one_face.append(mapping_new_to_old[element])
        result.append(sorted(one_face))   

    if len(not_connected_nodes) > 0:
       result += not_connected_nodes

    # result is the list of list
    # every element of the result is the list that contains face ids that we assume belong to the same person.
    output_elem = {"clusters": result}
    return output_elem, mapping_old_to_new

def find_threshold(model_name: str, distance_metric: str) -> float:
    """
    Retrieve pre-tuned threshold values for a model and distance metric pair

    Note that, cosine measure here computed as  1 - (a / (np.sqrt(b) * np.sqrt(c))).
    Args:
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        distance_metric (str): distance metric name. Options are cosine, euclidean
            and euclidean_l2.
    Returns:
        threshold (float): threshold value for that model name and distance metric
            pair. Distances less than this threshold will be classified same person.
    """

    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

    thresholds = {
        # "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}, # 2622d
        "VGG-Face": {
            "cosine": 0.68,
            "euclidean": 1.17,
            "euclidean_l2": 1.17,
        },  # 4096d - tuned with LFW
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
        "GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "euclidean_l2": 1.10},
        "InsightFace": {"cosine": 77}
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

    return threshold

      
