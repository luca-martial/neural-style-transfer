import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

# Load model from TF-Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2') 

# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Stylize function
def stylize(content_image, style_image):
    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    # Stylize image
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    return tensor_to_image(stylized_image)

# Add image examples for users    
joker = [["example_joker.jpeg"], ["example_polasticot1.jpeg"]]
paris = [["example_paris.jpeg"], ["example_vangogh.jpeg"]]
einstein = [["example_einstein.jpeg"], ["example_polasticot2.jpeg"]]
aristotle = [["example_aristotle.jpeg"], ["example_dali.jpeg"]]
avatar = [["example_avatar.jpeg"], ["example_polasticot3.jpeg"]]

# Customize interface
title = "Fast Neural Style Transfer using TF-Hub"
description = "Demo for neural style transfer using the pretrained Arbitrary Image Stylization model from TensorFlow Hub."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/1705.06830'>Exploring the structure of a real-time, arbitrary neural artistic stylization network</a></p>"
content_input = gr.inputs.Image(label="Content Image", source="upload")
style_input = gr.inputs.Image(label="Style Image", source="upload")

# Build and launch
iface = gr.Interface(fn=stylize, 
                     inputs=[content_input, style_input], 
                     outputs="image",
                     title=title,
                     description=description,
                     article=article,
                     examples=[joker, paris, einstein, aristotle, avatar])
iface.launch()