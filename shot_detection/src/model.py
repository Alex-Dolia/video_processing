import os

import cv2
import tensorflow as tf
import numpy as np


class TransNetV2:
    '''
    Model to make shot boundary detection, original source here: 
    https://github.com/soCzech/TransNetV2/blob/master/inference/transnetv2.py

    Params:
    ---------------------------------------------------------------------------
    model_dir: str, path to folder with pretrained model
    ---------------------------------------------------------------------------
    '''
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), 
                                     "../models/transnetv2-weights/")
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(
                    f"[TransNetV2] ERROR: {model_dir} is not a directory.")
            else:
                print(f"[TransNetV2] Using weights from {model_dir}.")

        self._input_size = (27, 48, 3)
        try:
            self._model = tf.saved_model.load(model_dir)
        except OSError as exc:
            raise IOError(
                f"[TransNetV2] It seems that files in {model_dir} are corrupted or missing. "
                f"Re-download them manually and retry. For more info, see: "
                f"https://github.com/soCzech/TransNetV2/issues/1#issuecomment-647357796"
                ) from exc

    def predict_raw(self, frames: np.ndarray) -> tuple:
        assert (len(frames.shape) == 5 and 
                frames.shape[2:] == self._input_size), \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
        frames = tf.cast(frames, tf.float32)

        logits, dict_ = self._model(frames)
        single_frame_pred = tf.sigmoid(logits)
        all_frames_pred = tf.sigmoid(dict_["many_hot"])

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames: np.ndarray) -> tuple:
        assert (len(frames.shape) == 4 and 
                frames.shape[1:] == self._input_size), \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            # return windows of size 100 where the first/last 25 frames are 
            # from the previous/next batch the first and last window must be 
            # padded by copies of the first and last frame of the video
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 
                                              if len(frames) % 50 != 0
                                                else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] +
                  [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out[np.newaxis]

        predictions = []

        for inp in input_iterator():
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((single_frame_pred.numpy()[0, 25:75, 0],
                                all_frames_pred.numpy()[0, 25:75, 0]))

            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(frames)), len(frames)
            ), end="")
        print("")

        single_frame_pred = np.concatenate([single_ for single_, all_ 
                                            in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ 
                                          in predictions])
        # remove extra padded frames
        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]

    def _convert_video_to_frames(self, video_fn: str, height: int, width: int
                                 ) -> np.array:
        '''Convert video to to frames'''
        cap = cv2.VideoCapture(video_fn)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        return np.array(frames)

    def predict_video(self, video_fn: str) -> tuple:
        video = self._convert_video_to_frames(video_fn=video_fn,
                                              height=self._input_size[0], 
                                              width=self._input_size[1])
        return (video, *self.predict_frames(video))

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5
                              ) -> np. array:
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions, 
                              line_width: float = 3):
        from PIL import Image, ImageDraw

        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, ic = frames.shape[1:]
        width = 25

        # pad frames so that length of the video is divisible by width
        # pad frames also by len(predictions) pixels in width in order to 
        # show predictions
        pad_with = width - len(frames) % width if len(frames
                                                      ) % width != 0 else 0
        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)),
                                  (0, 0)])

        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width

        img = frames.reshape([height, width, ih + 1, iw + len(predictions), 
                              ic])
        img = np.concatenate(np.split(
            np.concatenate(np.split(img, height), axis=2)[0], width
        ), axis=2)[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        # iterate over all frames
        for i, pred in enumerate(zip(*predictions)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

            # we can visualize multiple predictions per single frame
            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255

                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value), 
                              fill=tuple(color), width=line_width)
        return img
    
    def visualize_predictions_higher_quality(self, video_fn: str, 
                                             predictions,
                                             height: int = 100, 
                                             width: int = 160,
                                             line_width: int = 20):
        '''
        Create final image with boundaries in high quality for each picture

        Params:
        -----------------------------------------------------------------------
        video_fn: str, path to video
        predictions: tuple of arrays or np.array with boundary probabilities
        height: int, height of the image which will be extract from video: 
                     the height, the better quality
        width: int, width of the image which will be extract from video: 
                    the height, the better quality
        line_width: int, width of the line which is separate shots on the 
                         final picture
        -----------------------------------------------------------------------

        Returns:
        -----------------------------------------------------------------------
        image object
        -----------------------------------------------------------------------
        '''
        frames = self._convert_video_to_frames(video_fn=video_fn,
                                               height=height, 
                                               width=width)
        return self.visualize_predictions(frames=frames, 
                                          predictions=predictions, 
                                          line_width=line_width)
    
    @staticmethod
    def _get_video_duration(video_fn: str) -> float:
        '''get video duration in seconds'''
        cap = cv2.VideoCapture(video_fn)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        return duration
    
    def predictions_to_scenes_in_seconds(self,
                                         video_fn: str,
                                         predictions: np.ndarray, 
                                         threshold: float = 0.5) -> np. array:
        '''
        Create final image with boundaries in high quality for each picture

        Params:
        -----------------------------------------------------------------------
        video_fn: str, path to video
        predictions: np.array with boundary probabilities for frames
        threshold: float, threshold for boundary probabilities
        -----------------------------------------------------------------------

        Returns:
        -----------------------------------------------------------------------
        2D np.array with time_start and time_end for each shot
        -----------------------------------------------------------------------
        '''
        scenes = self.predictions_to_scenes(predictions=predictions,
                                            threshold=threshold)
        video_duration = self._get_video_duration(video_fn)
        return video_duration * (scenes/scenes[-1][1])
        
        