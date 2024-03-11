#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg19


def img_preprocess(image):
    max_dim = 512
    factor = max_dim / max(image.size)
    image = image.resize((round(image.size[0] * factor), round(image.size[1] * factor)), Image.Resampling.LANCZOS)
    im_array = img_to_array(image)
    im_array = np.expand_dims(im_array, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(im_array)
    return img

def img_preprocess_vgg16(image):
    max_dim = 512
    factor = max_dim / max(image.size)
    image = image.resize((round(image.size[0] * factor), round(image.size[1] * factor)), Image.Resampling.LANCZOS)
    im_array = img_to_array(image)
    im_array = np.expand_dims(im_array, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(im_array)
    return img

def img_preprocess_resnet50(image):
    max_dim = 512
    factor = max_dim / max(image.size)
    image = image.resize((round(image.size[0] * factor), round(image.size[1] * factor)), Image.Resampling.LANCZOS)
    im_array = img_to_array(image)
    im_array = np.expand_dims(im_array, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(im_array)
    return img

content_layers = ['block5_conv2']
number_content = len(content_layers)

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
number_style = len(style_layers)

content_layers_resnet50 = ['conv3_block4_out']
style_layers_resnet50 = ['conv1_relu',
                'conv2_block3_out',
                'conv3_block4_out',
                'conv4_block6_out',
                'conv5_block3_out']

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3 #Input dimension must be [1, height, width, channel] or [height, width, channel]
  
  
  # perform the inverse of the preprocessing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1] # converting BGR to RGB channel

  x = np.clip(x, 0, 255).astype('uint8')
  return x

def get_model_vgg19():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    return Model(vgg.input, model_outputs)

def get_model_vgg16():
    
    vgg=tf.keras.applications.vgg16.VGG16(include_top=False,weights='imagenet')
    vgg.trainable=False
    content_output=[vgg.get_layer(layer).output for layer in content_layers]
    style_output=[vgg.get_layer(layer).output for layer in style_layers]
    model_output= style_output+content_output
    return Model(vgg.input,model_output)


def get_model_resnet50():
    
    resnet =tf.keras.applications.ResNet50(include_top=False,weights='imagenet')
    resnet .trainable=False
    content_output=[resnet.get_layer(layer).output for layer in content_layers_resnet50]
    style_output=[resnet.get_layer(layer).output for layer in style_layers_resnet50]
    model_output= style_output+content_output
    return Model(resnet.input,model_output)

def get_style_loss(noise,target):
    gram_noise=gram_matrix(noise)
    #gram_target=gram_matrix(target)
    loss=tf.reduce_mean(tf.square(target-gram_noise))
    return loss

def get_content_loss(noise,target):
    loss = tf.reduce_mean(tf.square(noise-target))
    return loss

def get_features_resnet50(model, content_image, style_image):
    content_features = model(img_preprocess_resnet50(content_image))[len(style_layers_resnet50):]
    style_features = model(img_preprocess_resnet50(style_image))[:len(style_layers_resnet50)]
    return content_features, style_features

def get_features_vgg16(model, content_image, style_image):
    content_features = model(img_preprocess_vgg16(content_image))[len(style_layers):]
    style_features = model(img_preprocess_vgg16(style_image))[:len(style_layers)]
    return content_features, style_features

def get_features_vgg16(model, content_image, style_image):
    content_features = model(img_preprocess_vgg16(content_image))[len(style_layers):]
    style_features = model(img_preprocess_vgg16(style_image))[:len(style_layers)]
    return content_features, style_features

def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def compute_loss(model, loss_weights,image, gram_style_features, content_features):
    style_weight,content_weight = loss_weights #style weight and content weight are user given parameters
                                               #that define what percentage of content and/or style will be preserved in the generated image
    
    output=model(image)
    content_loss=0
    style_loss=0
    
    noise_style_features = output[:number_style]
    noise_content_feature = output[number_style:]
    
    weight_per_layer = 1.0/float(number_style)
    for a,b in zip(gram_style_features,noise_style_features):
        style_loss+=weight_per_layer*get_style_loss(b[0],a)
        
    
    weight_per_layer =1.0/ float(number_content)
    for a,b in zip(noise_content_feature,content_features):
        content_loss+=weight_per_layer*get_content_loss(a[0],b)
        
    style_loss *= style_weight
    content_loss *= content_weight
    
    total_loss = content_loss + style_loss
    
    
    return total_loss,style_loss,content_loss

def compute_grads(dictionary):
    with tf.GradientTape() as tape:
        all_loss=compute_loss(**dictionary)
        
    total_loss=all_loss[0]
    return tape.gradient(total_loss,dictionary['image']),all_loss

def run_style_transfer_vgg19(content_image, style_image, epochs=300, content_weight=1e3, style_weight=1e-2):
    model = get_model_vgg19()
    content_features, style_features = get_features(model, content_image, style_image)
    
    # Create the style gram matrices
    style_gram_matrix = [gram_matrix(feature) for feature in style_features]
    
    # Preprocess the content image and create a TensorFlow Variable
    noise = img_preprocess(content_image)
    noise = tf.Variable(noise, dtype=tf.float32)
    
    # Set up the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
    
    # Initialize variables to track the best result
    best_loss, best_img = float('inf'), None
    
    # Set up the weights for style and content loss
    loss_weights = (style_weight, content_weight)
    
    # Dictionary to pass to the compute_grads function
    dictionary = {
        'model': model,
        'loss_weights': loss_weights,
        'image': noise,
        'gram_style_features': style_gram_matrix,
        'content_features': content_features
    }
    
    # Normalization bounds for the image
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
   
    # Run the style transfer for the specified number of epochs
    for i in range(epochs):
        grads, all_loss = compute_grads(dictionary)
        total_loss, style_loss, content_loss = all_loss
        optimizer.apply_gradients([(grads, noise)])
        clipped = tf.clip_by_value(noise, min_vals, max_vals)
        noise.assign(clipped)
        
        # Update the best image if the total loss is improved
        if total_loss < best_loss:
            best_loss = total_loss
            best_img = deprocess_img(noise.numpy())

        # Optional: Show intermediate results
        if i % 5 == 0:
            plot_img = deprocess_img(noise.numpy())
            # Code to display intermediate images, if desired

    return best_img  # Return the final styled image



def run_style_transfer_vgg16(content_image, style_image, epochs=300, content_weight=1e3, style_weight=1e-2):
    model = get_model_vgg16()
    content_features, style_features = get_features_vgg16(model, content_image, style_image)
    
    # Create the style gram matrices
    style_gram_matrix = [gram_matrix(feature) for feature in style_features]
    
    # Preprocess the content image and create a TensorFlow Variable
    noise = img_preprocess_vgg16(content_image)
    noise = tf.Variable(noise, dtype=tf.float32)
    
    # Set up the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
    
    # Initialize variables to track the best result
    best_loss, best_img = float('inf'), None
    
    # Set up the weights for style and content loss
    loss_weights = (style_weight, content_weight)
    
    # Dictionary to pass to the compute_grads function
    dictionary = {
        'model': model,
        'loss_weights': loss_weights,
        'image': noise,
        'gram_style_features': style_gram_matrix,
        'content_features': content_features
    }
    
    # Normalization bounds for the image
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
   
    # Run the style transfer for the specified number of epochs
    for i in range(epochs):
        grads, all_loss = compute_grads(dictionary)
        total_loss, style_loss, content_loss = all_loss
        optimizer.apply_gradients([(grads, noise)])
        clipped = tf.clip_by_value(noise, min_vals, max_vals)
        noise.assign(clipped)
        
        # Update the best image if the total loss is improved
        if total_loss < best_loss:
            best_loss = total_loss
            best_img = deprocess_img(noise.numpy())

        # Optional: Show intermediate results
        if i % 5 == 0:
            plot_img = deprocess_img(noise.numpy())
            # Code to display intermediate images, if desired

    return best_img  # Return the final styled image

def run_style_transfer_resnet50(content_image, style_image, epochs=800, content_weight=1e3, style_weight=1e-2):
    model = get_model_resnet50()
    content_features, style_features = get_features_resnet50(model, content_image, style_image)
    
    # Create the style gram matrices
    style_gram_matrix = [gram_matrix(feature) for feature in style_features]
    
    # Preprocess the content image and create a TensorFlow Variable
    noise = img_preprocess_resnet50(content_image)
    noise = tf.Variable(noise, dtype=tf.float32)
    
    # Set up the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
    
    # Initialize variables to track the best result
    best_loss, best_img = float('inf'), None
    
    # Set up the weights for style and content loss
    loss_weights = (style_weight, content_weight)
    
    # Dictionary to pass to the compute_grads function
    dictionary = {
        'model': model,
        'loss_weights': loss_weights,
        'image': noise,
        'gram_style_features': style_gram_matrix,
        'content_features': content_features
    }
    
    # Normalization bounds for the image
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
   
    # Run the style transfer for the specified number of epochs
    for i in range(epochs):
        grads, all_loss = compute_grads(dictionary)
        total_loss, style_loss, content_loss = all_loss
        optimizer.apply_gradients([(grads, noise)])
        clipped = tf.clip_by_value(noise, min_vals, max_vals)
        noise.assign(clipped)
        
        # Update the best image if the total loss is improved
        if total_loss < best_loss:
            best_loss = total_loss
            best_img = deprocess_img(noise.numpy())

        # Optional: Show intermediate results
        if i % 5 == 0:
            plot_img = deprocess_img(noise.numpy())
            # Code to display intermediate images, if desired

    return best_img  # Return the final styled image


st.title('Artistic Style Transfer')
st.write("Upload a food or drink image and an artistic image to apply style transfer.")

# Uploaders for content and style images
content_file = st.file_uploader("Upload Food/Drinks Image...", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("Upload Artistic Image...", type=["jpg", "png", "jpeg"])

if content_file and style_file:
    content_image = Image.open(content_file)
    style_image = Image.open(style_file)

    st.image(content_image, caption='Content Image', use_column_width=True)
    st.image(style_image, caption='Style Image', use_column_width=True)

    model_option = st.selectbox('Choose a style transfer model:', ('VGG19', 'VGG16', 'ResNet50'))

    if st.button('Apply Style Transfer'):
        st.write("Generating styled image...")
        if model_option == 'VGG19':
            styled_image = run_style_transfer_vgg19(content_image, style_image)
        elif model_option == 'VGG16':
            styled_image = run_style_transfer_vgg16(content_image, style_image)
        elif model_option == 'ResNet50':
            styled_image = run_style_transfer_resnet50(content_image, style_image)


        st.image(styled_image, caption='Styled Image', use_column_width=True)


