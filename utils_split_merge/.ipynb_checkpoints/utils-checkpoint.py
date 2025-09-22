import numpy as np
import pandas as pd

def csm(A,B):
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)

def inverse_cosine_similarity_fn(emb_quality_movie_tracker_output, face_embedding_name = 'embedding'):
    face_embeddings = [elem[face_embedding_name] for elem in emb_quality_movie_tracker_output]
    n_track_ids = [str(elem["n"]) + "_" + str(elem["track_id"]) 
                   for elem in emb_quality_movie_tracker_output]
   
    face_embeddings = np.array(face_embeddings)
          
    cosine_similarities = csm(face_embeddings, face_embeddings)

    cosine_similarities_df = pd.DataFrame(cosine_similarities, columns = n_track_ids)

    cosine_similarities_df["first"] = n_track_ids
    unpivot_cosine_similarities_df = pd.melt(cosine_similarities_df, id_vars="first", var_name = "second", value_vars=n_track_ids)

    unpivot_cosine_similarities_df["first_n"] = unpivot_cosine_similarities_df["first"].apply(lambda x: int(x.split("_")[0]))
    unpivot_cosine_similarities_df["first_track_id"] = unpivot_cosine_similarities_df["first"].apply(lambda x: int(x.split("_")[1]))

    unpivot_cosine_similarities_df["second_n"] = unpivot_cosine_similarities_df["second"].apply(lambda x: int(x.split("_")[0]))
    unpivot_cosine_similarities_df["second_track_id"] = unpivot_cosine_similarities_df["second"].apply(lambda x: int(x.split("_")[1]))
 
    unpivot_cosine_similarities_df["inverse_cosine"] = 1 - unpivot_cosine_similarities_df["value"]

    unpivot_cosine_similarities_df = unpivot_cosine_similarities_df[["first", "second", "first_n", "first_track_id", "second_n", "second_track_id", "inverse_cosine"]].copy()

    return unpivot_cosine_similarities_df

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
    # n  track_id   height   width   confidence   face_quality  #:
    # 0         1      310     225     0.733119       0.578306  1

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
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

    return threshold

      
