import os
os.environ["TRANSFORMERS_NO_METAL"] = "1"
os.environ["ACCELERATE_DISABLE_METATENSOR"] = "1"
os.environ["TRANSFORMERS_NO_META_DEVICE"] = "1"
import pandas as pd
import transformers
import streamlit as st
import html, re
import torch
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertModel
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, SeparableConv1D, LayerNormalization, Dense, GlobalAveragePooling1D, Multiply, Add, Conv1D, Dropout
import pickle
print("Streamlit:", st.__version__)
print("TensorFlow:", tf.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)


# ----------------------
# Custom Layers
# ----------------------
class ConvMixerBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, dropout_rate=0.1, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.depthwise_pw = SeparableConv1D(filters, kernel_size, padding='same',
                                            dilation_rate=dilation_rate, activation='relu')
        self.layernorm = LayerNormalization()
        self.dropout = Dropout(dropout_rate)
        self.proj = None

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        out_ch = int(self.depthwise_pw.filters)
        if in_ch != out_ch:
            self.proj = Conv1D(out_ch, 1, padding='same')
        super().build(input_shape)

    def call(self, inputs):
        x = self.depthwise_pw(inputs)
        x = self.layernorm(x)
        x = self.dropout(x)
        res = inputs
        if self.proj is not None:
            res = self.proj(res)
        return Add()([x, res])

class ChannelSelfAttention(tf.keras.layers.Layer):
    def __init__(self, filters=128, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.global_avg_pool = GlobalAveragePooling1D()
        self.dense1 = Dense(max(4, filters//reduction), activation='relu')
        self.dense2 = Dense(filters, activation='sigmoid')

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.expand_dims(x, axis=1)
        return Multiply()([inputs, x])

# ----------------------
# Load Model & Scaler
# ----------------------
try:
    model = load_model(
        'my_model.h5',
        custom_objects={'ConvMixerBlock': ConvMixerBlock, 'ChannelSelfAttention': ChannelSelfAttention}
    )
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_mean = scaler['mean']
    X_std = scaler['std']
    print("Scaler loaded successfully!")
except Exception as e:
    print("Error loading scaler:", e)



# ----------------------
# Load DistilBERT
# ----------------------
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import html, re
import numpy as np

# ------------------------------------------
# FIX for meta tensor issue in transformers
# ------------------------------------------


# ----------------------
# Device & Model
# ----------------------
import torch
from transformers import DistilBertTokenizer, DistilBertModel

@st.cache_resource
def load_bert():
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    bert_model = DistilBertModel.from_pretrained(
        "distilbert-base-uncased",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        device_map=None
    )

    bert_model.eval()
    return tokenizer, bert_model


tokenizer, bert_model = load_bert()





# ----------------------
# Preprocessing & Embedding
# ----------------------
def preprocess(text):
    temp = html.unescape(text)
    temp = re.sub(r'@[A-Za-z0-9_]+','', temp)
    temp = re.sub(r'https?://[A-Za-z0-9./]+','', temp)
    temp = re.sub(r'[^a-zA-Z\s]','', temp)
    temp = temp.lower()
    temp = re.sub(' +',' ', temp)
    return temp.strip()
def get_embedding(text):
    text = preprocess(text)
    inputs = tokenizer(
        [text],
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)  # (1,768)
    embedding = embedding.detach().cpu().numpy().astype(np.float32)
    return embedding

  # let torch decide CPU placement





# ----------------------
# Prediction Function
# ----------------------
def predict_suicide(text):
    emb = get_embedding(text)                  # (1, 768)
    emb = (emb - X_mean) / X_std               # normalize
    emb = emb.reshape(1, 768, 1)               # shape (1, 768, 1)
    print("Embedding shape:", emb.shape)       # debug

    # Check if model is loaded
    if model is None:
        raise ValueError("Keras model not loaded properly!")

    pred_prob = model.predict(emb)[0][0]
    label = "Suicide" if pred_prob > 0.1 else "Not Suicide"
    return label, pred_prob



# ----------------------
# Streamlit App
# ----------------------
st.title("Suicide Detection App")
st.write("Enter a sentence, and the model will predict if it indicates suicide risk.")

user_input = st.text_area("Enter Text Here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence to predict.")
    else:
        label, prob = predict_suicide(user_input)
        st.success(f"Prediction: **{label}**")
        st.info(f"Probability: {prob:.4f}")
