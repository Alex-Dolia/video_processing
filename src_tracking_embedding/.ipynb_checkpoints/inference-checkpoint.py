import torch, os, json, io, cv2, time, numpy as np
import base64
import os

import cv2
import os
from PIL import Image
from deepface_ad.facenet_ad import FaceNet512dClient

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...", os.getcwd())
    model = FaceNet512dClient(model_dir)
    return model

def input_fn(request_body, request_content_type):
    """
    Preprocess the incoming request.

    Parameters:
    request_body (str): The body of the incoming request.
    request_content_type (str): The content type of the incoming request.

    Returns:

    """
    # Check the content type
    if request_content_type == 'application/json':
        # Parse the JSON object
        request_data = json.loads(request_body)
        return request_data

    raise ValueError(f"Unsupported content type: {request_content_type}")
    
    
      
def predict_fn_ORIGINAL(input_data, model):
    movie_tracker_output = input_data["movie_tracker_output"].copy()
    nufs = input_data["nufs"]
    print("Executing predict_fn from inference.py ...")
    output = []
    embeddings = []
    nnn = []
        
    for i, elem in enumerate(movie_tracker_output):
        if i % 10 == 0:
           print("i: ", i) 
        face = elem["face"].copy()

        rgb_img = np.asarray(Image.fromarray(np.uint8(face)).convert('RGB')).copy()    
        # rgb to bgr
        norm_brg_img = rgb_img[:, :, ::-1] / 255.0          
        
        target_size = model.input_shape
        
        norm_brg_img = model.resize_image(img=norm_brg_img, target_size = (target_size[1], target_size[0]))
        #elem["norm_brg_img"] = norm_brg_img.copy()
        embedding = model.forward(norm_brg_img)
        elem["embedding"] = embedding
        output.append(elem)
                
    del model   
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
        
    processed_input_data = {"movie_tracker_output": output, "nufs": nufs}
    return processed_input_data   


from tqdm import tqdm  # Import tqdm for progress tracking

def predict_fn(input_data, model, embedding_type = "FaceNet512"):
    movie_tracker_output = input_data["movie_tracker_output"].copy()
    nufs = input_data["nufs"]

    output = []

    if embedding_type == "InsightFace":
       print("!!!!!!!!!!!!!!!!!!!!!!!!!I do embedding_type InsightFace!!!!!!!!!!!!!!!!!!!!!!!!!") 
       import insightface
       from insightface.app import FaceAnalysis
       import cv2
       model_pack_name = 'buffalo_l'
       app = FaceAnalysis(name=model_pack_name)
       app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Initialize progress bar
    for elem in tqdm(movie_tracker_output, desc="Processing images", unit="img"):
        embedding = None
        if embedding_type == "InsightFace":
           x1, y1, x2, y2 = elem["bounding_box"].copy() 
           x1, y1, x2, y2 = int(x1)-10, int(y1)-10, int(x2)+10, int(y2)+10
           x1 = max(0, x1)
           y1 = max(0, y1) 
           face = elem["frame"][y1:y2, x1:x2].copy()
           print("x1, y1, x2, y2: ", x1, y1, x2, y2) 
           print("face.shape: ", face.shape) 
            
           detected_faces = app.get(face) 
           if len(detected_faces) > 0:  
              embedding = np.array(detected_faces[0].normed_embedding)
              print("!!!_________________________!!! np.shape(embedding): ", np.shape(embedding)) 
        else:
           face = elem["face"].copy() 
           rgb_img = np.asarray(Image.fromarray(np.uint8(face)).convert('RGB')).copy()    
           # Convert RGB to BGR and normalize
           brg_img = rgb_img[:, :, ::-1].copy() 
           norm_brg_img = brg_img / 255.0          
           target_size = model.input_shape
           norm_brg_img = model.resize_image(img=norm_brg_img, target_size=(target_size[1], target_size[0]))
           embedding = model.forward(norm_brg_img)
   
        if embedding is not None:
           del elem["frame"] 
           elem["embedding"] = embedding
           output.append(elem)
                
    del model   
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
        
    processed_input_data = {"movie_tracker_output": output, "nufs": nufs}
    return processed_input_data




def predict_fn_ANDREW(input_data, model):
    movie_tracker_output = input_data["movie_tracker_output"].copy()
    nufs = input_data["nufs"]
    print("Executing predict_fn from inference.py ...")
    output = []
    embeddings = []
    nnn = []
    
    #########################################################################################################
    #########################################################################################################
    import insightface
    from insightface.app import FaceAnalysis
    import cv2
    model_pack_name = 'buffalo_l'
    app = FaceAnalysis(name=model_pack_name)
    #app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' if you have a GPU
    app.prepare(ctx_id=0, det_size=(640, 640))
    #########################################################################################################
    #########################################################################################################
    
    for i, elem in enumerate(movie_tracker_output):
        if i % 10 == 0:
           print("i: ", i) 
        face = elem["face"].copy()

        rgb_img = np.asarray(Image.fromarray(np.uint8(face)).convert('RGB')).copy()
        
        ######################
        ######################
        print("!!! elem.keys(): ", elem.keys()) 
        cv2_img = np.asarray(Image.fromarray(np.uint8(elem["frame"]))).copy()
        
        frame_id = elem["n"] # REMOVE
        os.makedirs("merge_predict_fn/frames/", exist_ok=True)
        print("!_!!!!!!!!!!!!!!!_! type(face): ", type(cv2_img), cv2_img.shape)
        cv2.imwrite(f"merge_predict_fn/frames/{frame_id:05d}.png", cv2_img)
        nnn.append(frame_id)
        ######################
        ######################
        ################################################################
        faces = app.get(cv2_img)
        print("!!! befor embedding np.shape(faces): ", ", frame_id: ", frame_id)
        for face in faces:
            embedding = face.embedding
            print("np.shape(embedding): ", np.shape(embedding))
            embeddings.append(list(embedding[0]))
        ################################################################
        
        # rgb to bgr
        norm_brg_img = rgb_img[:, :, ::-1] / 255.0          
        
        target_size = model.input_shape
        
        norm_brg_img = model.resize_image(img=norm_brg_img, target_size = (target_size[1], target_size[0]))
        #elem["norm_brg_img"] = norm_brg_img.copy()
        embedding = model.forward(norm_brg_img)
        elem["embedding"] = embedding
        output.append(elem)
        
        
    from sklearn.cluster import DBSCAN
    clt = DBSCAN(eps=0.4,min_samples=1,metric="cosine",n_jobs=-1)
    embeddings = np.array(embeddings)
    print("!!!!!!!!!!!!!!!!!!!!!!!! embeddings: ", embeddings.shape)
    clt.fit(embeddings)
    labelIDs = np.unique(clt.labels_)
    labelIDs_df =pd.DataFrame({"labelIDs": labelIDs, "n": nnn}).to_csv("merge_predict_fn/labelIDs.csv")
    numUniqueFaces = len(np.where(labelIDs >= -1)[0])   
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("numUniqueFaces", numUniqueFaces)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        
    del model   
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
        
    processed_input_data = {"movie_tracker_output": output, "nufs": nufs}
    return processed_input_data 

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
    return json.dumps(prediction_output, cls = NumpyEncoder)