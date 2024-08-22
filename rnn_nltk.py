import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Load the data and preprocess it
df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
df = df.dropna(subset=['Review Text']) 

# Convert positive 1 and negative 0
df['Sentiment'] = np.where(df['Rating'] > 3, 1, 0)

X = df['Review Text'].values
y = df['Sentiment'].values

# Split test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences of integers
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length across all inputs
X_train_pad = pad_sequences(X_train_seq)
X_test_pad = pad_sequences(X_test_seq)

# Build the RNN model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64),
    SimpleRNN(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test))


def predict_sentiment(review):
    review_seq = tokenizer.texts_to_sequences([review])
    review_pad = pad_sequences(review_seq, maxlen=X_train_pad.shape[1]) 
    prediction = model.predict(review_pad)
    return 'Positive' if prediction > 0.5 else 'Negative'

# Streamlit app
st.title("Sentiment Analysis on Reviews")
st.write("Enter a review to predict whether it's positive or negative.")

# Text input
user_input = st.text_area("Review Text", "")

if st.button("Predict Sentiment"):
    if user_input:
        result = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: {result}")
    else:
        st.write("Please enter a review text.")
model.save('rnn.h5')