import os
import json
import datetime
import streamlit as st
import numpy as np
import re
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Prepare dataset
def prepare_data(intents):
    tags = []
    patterns = []
    responses = {}
    for intent in intents:
        tag = intent["tag"]
        tags.append(tag)
        responses[tag] = intent["responses"]
        patterns.extend([(p, tag) for p in intent["patterns"]])
    return patterns, responses

patterns, responses = prepare_data(intents)

# Separate features (patterns) and labels (tags)
X = [p[0] for p in patterns]  # Patterns
y = [p[1] for p in patterns]  # Tags

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

X = [preprocess_text(x) for x in X]

# Encode labels
tag_to_index = {tag: index for index, tag in enumerate(set(y))}
index_to_tag = {index: tag for tag, index in tag_to_index.items()}
y_encoded = [tag_to_index[tag] for tag in y]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = max(len(x) for x in X_train_seq)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Build Transformer model
def create_transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64)(inputs)
    
    # Transformer block
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)  # Residual connection
    x = Dropout(0.1)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# Create and compile the model
model = create_transformer_model((max_length,), len(tag_to_index))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, np.array(y_train), epochs=10, batch_size=8, validation_split=0.1)

# Evaluate the model
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes))

# Chatbot
def chatbot(input_text):
    input_text = preprocess_text(input_text)
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_pad = pad_sequences(input_seq, maxlen=max_length, padding='post')
    predicted_index = np.argmax(model.predict(input_pad), axis=1)[0]
    predicted_tag = index_to_tag[predicted_index]
    return random.choice(responses[predicted_tag])

def log_conversation(user_input, response):
    """Log the conversation to session state."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    st.session_state.conversation_history.append((user_input, response, timestamp))

def main():
    st.markdown("""
        <style>
        .chat-container {
            background-color: #f0f2f5;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .user-input {
            font-weight: bold;
            color: #007bff;
        }
        .chatbot-response {
            font-weight: bold;
            color: #28a745;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("EDVERTO: The Chatbot")
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    menu = ["Home", "Conversation History", "About EDVERTO"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to Edverto. Start your conversation below!")
        user_input = st.text_input("You:", key="user_input")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, key="chatbot_response")

            log_conversation(user_input, response)
            if response.lower() in ["goodbye", "bye"]:
                st.write("Thank you for chatting. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        if "conversation_history" in st.session_state:
            for user_input, response, timestamp in st.session_state.conversation_history:
                st.text(f"User:  {user_input}")
                st.text(f"Chatbot: {response}")
                st.text(f"Timestamp: {timestamp}")
                st.markdown("---")
        else:
            st.write("No conversation history found.")

    elif choice == "About EDVERTO":
        st.write("This chatbot uses a Transformer model to classify intents and respond appropriately.")
        st.write("Project Overview:")
        st.write(
            "The chatbot employs a Transformer-based model for intent recognition, which allows it to learn from the data and respond quickly and accurately."
        )

if __name__ == "__main__":
    main()