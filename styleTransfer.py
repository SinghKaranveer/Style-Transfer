import numpy as np
from PIL import Image
import requests
import io
import time
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import IPython.display

# ML imports
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.keras import layers
from tensorflow.keras import backend

MAX_DIM = 700

content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]
num_style_layers = len(style_layers)
num_content_layers = len(content_layers)

def load_image(path):
    max_dim = 1024
    img = Image.open(path)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize((MAX_DIM, MAX_DIM))
    return img

def imshow(img):
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()

def processImage(img):
    processed_img = np.asarray(img, dtype = "float32")
    processed_img = np.expand_dims(processed_img, axis = 0)
    processed_img[:, :, :, 0] -= 103.939
    processed_img[:, :, :, 1] -= 116.779
    processed_img[:, :, :, 2] -= 123.68
    processed_img = processed_img[:, :, :, ::-1]
    return processed_img

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def deprocessImage(processed_img):
    output = processed_img.copy()
    if len(output.shape) == 4:
        output = np.squeeze(output, 0)
    output[:, :, 0] += 103.939
    output[:, :, 1] += 116.779
    output[:, :, 2] += 123.68
    output = output[:, :, ::-1]
    output = np.clip(output, 0, 255).astype('uint8')
    return output

def load_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    return models.Model(vgg.input, model_outputs)

def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))

def get_feature_representations(model, content_path, style_path):

    content_image = load_image(content_path)
    content_image = processImage(content_image)
  
    style_image = load_image(style_path)
    style_image = processImage(style_image)
  
    style_outputs = model(style_image)
    content_outputs = model(content_image)
  
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features



def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    model_outputs = model(init_image)
  
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
  
    style_score = 0
    content_score = 0


    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
      style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score 
    return loss, style_score, content_score

def compute_grads(cfg):
    with tf.GradientTape() as tape: 
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


def run_style_transfer(content_path, 
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3, 
                       style_weight=1e-2): 

  model = load_model() 
  for layer in model.layers:
    layer.trainable = False
  
  style_features, content_features = get_feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
  init_image = load_image(content_path)
  init_image = processImage(init_image)

  init_image = tfe.Variable(init_image, dtype=tf.float32)
  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)


  iter_count = 1
  

  best_loss, best_img = float('inf'), None
  
  loss_weights = (style_weight, content_weight)
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
  }
    
  num_rows = 2
  num_cols = 5
  display_interval = num_iterations/(num_rows*num_cols)
  start_time = time.time()
  global_start = time.time()
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  
  imgs = []
  for i in range(num_iterations):
    print(i)
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time() 
    
    if loss < best_loss: 
      best_loss = loss
      best_img = deprocessImage(init_image.numpy())

    if i % display_interval== 0:
      start_time = time.time()
      
      plot_img = init_image.numpy()
      plot_img = deprocessImage(plot_img)
      imgs.append(plot_img)
      IPython.display.clear_output(wait=True)
      IPython.display.display_png(Image.fromarray(plot_img))
      print('Iteration: {}'.format(i))        
      print('Total loss: {:.4e}, ' 
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '
            'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
  print('Total time: {:.4f}s'.format(time.time() - global_start))
  IPython.display.clear_output(wait=True)
  plt.figure(figsize=(14,4))
  for i,img in enumerate(imgs):
      plt.subplot(num_rows,num_cols,i+1)
      plt.imshow(img)
      plt.xticks([])
      plt.yticks([])
      
  return best_img, best_loss

def show_results(best_img, content_path, style_path, show_large_final=True):
  plt.figure(figsize=(10, 5))
  content = load_image(content_path) 
  style = load_image(style_path)

  plt.subplot(1, 2, 1)
  imshow(content)

  plt.subplot(1, 2, 2)
  imshow(style)

  if show_large_final: 
    plt.figure(figsize=(10, 10))

    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()


if __name__ == "__main__":
    style_path = "./images/abstract.jpg"
    content_path = "./images/HardySimran.jpg"
    tf.enable_eager_execution()
    best, best_loss = run_style_transfer(content_path, style_path, num_iterations=1000)
    show_results(best, content_path, style_path)
    print("finished")
 