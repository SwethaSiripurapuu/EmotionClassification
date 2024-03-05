import pandas as pd
import tensorflow as tf
import datasets
from transformers import AutoTokenizer
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Constants
TRAIN_FILE = 'train.csv'
DEV_FILE = 'dev.csv'
TEST_FILE = 'test-in.csv'

MODEL_PATH = 'model'
EPOCHS = 6
BATCH_SIZE = 32

# Define a function to tokenize text
def tokenize_function(examples, tokenizer, max_length=64):
    return tokenizer(examples.tolist(), truncation=True, padding=True, max_length=max_length)

# Define the neural network model
def build_model(tokenizer_vocab_len, output_units=7, lstm_units=128, dense_units=128, dropout_rate=0.4):
    model = models.Sequential([
        layers.Input(shape=(64,), dtype='int32'), 
        layers.Embedding(input_dim=tokenizer_vocab_len, output_dim=128),
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),
        layers.GlobalMaxPooling1D(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(output_units, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)])
    return model
    

# Function for prediction
def predict(model_path, input_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Load the data for prediction
    df = pd.read_csv(input_path)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Tokenize the input data
    encodings = tokenizer(df['text'].tolist(), truncation=True, padding=True, max_length=64, return_tensors="tf")

    # Create TensorFlow dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices(encodings['input_ids']).batch(BATCH_SIZE)

    # Generate predictions from model
    predictions = model.predict(tf_dataset)
    prediction_labels = (predictions > 0.5).astype(int)

    # Assign predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = prediction_labels

    # Write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(method='zip', archive_name='submission.csv'))

def main():
    # Load datasets
    train_df = pd.read_csv(TRAIN_FILE)
    dev_df = pd.read_csv(DEV_FILE)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Tokenize text
    train_encodings = tokenize_function(train_df['text'], tokenizer)
    dev_encodings = tokenize_function(dev_df['text'], tokenizer)

    # Convert labels to float32
    train_labels = train_df.iloc[:, 1:].values.astype('float32')
    dev_labels = dev_df.iloc[:, 1:].values.astype('float32')

    # Build model
    model, callbacks = build_model(len(tokenizer.vocab))
    model.summary()

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_encodings['input_ids'], train_labels)).batch(BATCH_SIZE)
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_encodings['input_ids'], dev_labels)).batch(BATCH_SIZE)

    # Train model
    history = model.fit(
        train_dataset, 
        validation_data=dev_dataset, 
        epochs=EPOCHS, 
        callbacks=callbacks + [tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True)]
    )
    # Predict and evaluate on dev dataset
    predict(MODEL_PATH, TEST_FILE)

if __name__ == "__main__":
    main()
