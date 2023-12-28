import numpy as np
import pandas as pd
import os
import cv2
import keras
from typing import  Tuple

IMG_SIZE = 224 #frame size
MAX_SEQ_LENGTH = 20 #max number of frame to collect per videos
NUM_FEATURES = 2048 
EPOCHS = 40

##function to read all the videos and get collect some caracteristics
def reading_data(path: str ) -> pd.DataFrame:
    """
    Opens the video folders located in base_path and returns a DataFrame.
    The DataFrame lists all the video file paths, frame count, fps, and duration.

    Parameters:
    - path (str): Path to the video folders in base_path.

    Returns:
    - pd.DataFrame: DataFrame with columns ['Label', 'video_path', 'Frame_count', 'fps', 'Duration (s)'].
    """
    result = []
    for folders, subfolders, files in os.walk(path):
        if folders == path:
            continue
        else:
            try:
                for file in files:
                    video = cv2.VideoCapture(folders + "/" + file)  # Read the video file using OpenCV
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Count the number of frames
                    fps = video.get(cv2.CAP_PROP_FPS)
                    video_length = frame_count / fps
                    result.append((folders.split('/')[-1], str(folders + "/" + file), frame_count, fps, video_length))
            except Exception as e:
                continue
    return pd.DataFrame(data=result, columns=['Label', 'video_path', 'Frame_count', 'fps', 'Duration (s)'])


#function to crop frames
def crop_center_square(frame:np.ndarray)->np.ndarray:
    """
    Crops a frame from the center to generate a square frame with dimensions equal to the minimum
    dimension of the original frame (height or width).

    Parameters:
    frame (numpy.ndarray): The input frame to be cropped.

    Returns:
    numpy.ndarray: The cropped square frame.
    """
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


#function to load a video
def load_video(path: str, max_frames: int = 20, resize: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    """
    Loads a video from the given path, extracts frames, crops, resizes, and converts them to an array.

    Parameters:
    path (str): The path to the video file.
    max_frames (int): The maximum number of frames to extract (default is 20).
    resize (Tuple[int, int]): The desired dimensions for resizing frames (default is (IMG_SIZE, IMG_SIZE) ie (224,224)).

    Returns:
    np.ndarray: An array containing the processed video frames.
    """
    cap = cv2.VideoCapture(path) #open the video stream to collect frame
    frames = []  # List to contain the frames

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            #crop the collected frame
            frame = crop_center_square(frame)
            #resize it to (IMG_SIZE,IMG_SIZE)
            frame = cv2.resize(frame, resize)
            #change the colors channels
            frame = frame[:, :, [2, 1, 0]]
            #append the frame to the frame list
            frames.append(frame)

            if len(frames) == max_frames:
                break  # Capture and resize frames until reaching max_frames
    finally:
        cap.release() #release the video stream
    return np.array(frames)


def prepare_single_video(frames: np.ndarray,feature_extractor) -> np.ndarray:
    """
    Extracts frame features and masks for a single video.

    Parameters:
    - frames (np.ndarray): Array of video frames.

    Returns:
    - np.ndarray: Extracted frame features and masks.
    """
    frame_features =  np.array(list(map(feature_extractor.predict,frames)))

    return frame_features


def sequence_prediction(path,model,label_processor):
    class_vocab = label_processor.get_vocabulary()
    frames = load_video(path)
    frame_features = prepare_single_video(frames)
    probabilities = model.predict(frame_features)[0]
    max_label = class_vocab[np.argsort(probabilities)[-1]]
    for i in np.argsort(probabilities)[::-1]:
        print(f"{class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return max_label


#build features object to extract key features from each frames

def build_feature_extractor(IMG_SIZE):
    """This function will build feature extractor model based on
     a pre-trained CNN namely InceptionV3
    to extract feature from each frames

    Returns:
    Model: Keras feature extractor Model"""

    #create the feature extractor based on InceptionV3
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False, #only include the Conv and pooling layers
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


MODEL_dict = {"GRU":keras.layers.GRU,"LSTM":keras.layers.LSTM}
def get_sequence_model(model: str, label_processor,MAX_SEQ_LENGTH=MAX_SEQ_LENGTH,NUM_FEATURES=NUM_FEATURES, models_dict=MODEL_dict)->keras.Model:
    """
    Generates a sequence model based on the specified architecture.

    Parameters:
    - model (str): Name of the model architecture t be built.
    - label_processor: Label preprocessor object.
    - models_dict: Dictionary containing available model architectures.

    Returns:
    - Model: Compiled sequence model for training.
    """
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    #mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    #building the Model
    if model == "BILSTM":
          x = keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=True))(frame_features_input)
          x = keras.layers.Bidirectional(keras.layers.LSTM(8))(x)
    else:
          x = models_dict[model](16, return_sequences=True)(frame_features_input)
          x = models_dict[model](8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    seq_model = keras.Model(frame_features_input, output)

    seq_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return seq_model


#training function


def run_experiment(model: str,label_processor,X_train,y_train,X_test,y_test, epochs: int = EPOCHS) -> Tuple[keras.callbacks.History, keras.Model]:
    """
    Trains a sequence model based on the specified architecture and runs an experiment.

    parameters:
    - model (str): Name of the model architecture.
    - epochs (int): Number of epochs for training (default: EPOCHS).

    Returns:
    - Tuple[keras.callbacks.History, keras.Model]: Training history and trained model.
    """
    filepath =  "models/"+model + "_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=0
    )

    seq_model = get_sequence_model(model,label_processor)

    history = seq_model.fit(
        X_train,
        y_train,
        validation_split=0.3,
        epochs=epochs,
        callbacks=[checkpoint],
    )


    seq_model.load_weights(filepath)
    _, acc = seq_model.evaluate(X_test,y_test)
    print(f"Test accuracy: {round(acc * 100, 2)}%")

    return history, seq_model
