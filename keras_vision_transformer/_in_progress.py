

import os
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def padding_3d(size_before, size_after):
    pad_dim0_left = (size_after[0] - size_before[0]) // 2
    pad_dim1_left = (size_after[1] - size_before[1]) // 2
    pad_dim2_left = (size_after[2] - size_before[2]) // 2

    pad_dim0_right = (size_after[0] - size_before[0]) - pad_dim0_left
    pad_dim1_right = (size_after[1] - size_before[1]) - pad_dim1_left
    pad_dim2_right = (size_after[2] - size_before[2]) - pad_dim2_left
    
    return ((pad_dim0_left, pad_dim0_right), 
            (pad_dim1_left, pad_dim1_right), 
            (pad_dim2_left, pad_dim2_right))




class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(filters=embed_dim, kernel_size=patch_size, 
                                        strides=patch_size, padding="VALID",)
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(input_dim=num_tokens, output_dim=self.embed_dim)
        self.positions = tf.range(0, num_tokens, 1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

# 3-D SwinT
# --------------------------------------------------------------------- #
def window_partition_3d(x, window_size):
    _, time, height, width, channels = x.shape
    patch_num_t = time // window_size
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    
    x = tf.reshape(x, (-1, patch_num_t, window_size, patch_num_y, window_size, patch_num_x, window_size, channels,))
    x = tf.transpose(x, perm=[0, 1, 3, 5, 2, 4, 6, 7])
    windows = tf.reshape(x, (-1, window_size, window_size, window_size, channels))
    return windows


def window_reverse_3d(windows, window_size, time, height, width, channels):
    patch_num_t = time // window_size
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    
    x = tf.reshape(windows, (-1, patch_num_t, patch_num_y, patch_num_x, window_size, window_size, window_size, channels,))
    x = tf.transpose(x, perm=[0, 1, 4, 2, 5, 3, 6, 7])
    x = tf.reshape(x, (-1, time, height, width, channels))
    return x

# SwinT shift-window attention
class WindowAttention_3d(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs,):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

        num_window_elements = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.keras.initializers.Zeros(),
            name='SwinT_WindowAttention',
            trainable=True,)

        coords_t = np.arange(self.window_size[0])
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        
        coords_matrix = np.meshgrid(coords_t, coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        
        coords_flatten = coords.reshape(3, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        
        relative_coords = relative_coords.transpose([1, 2, 0])
        
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        
        relative_coords[:, :, 0] *= (2*self.window_size[1] - 1) * (2*self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        
        relative_position_index = relative_coords.sum(-1)
        # <-------- flatten pos_index when gather from the table
        
        self.relative_position_index = tf.Variable(
            initial_value=relative_position_index,
            shape=relative_position_index.shape,
            dtype="int32",
            trainable=False,)

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, (-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=[0, 1, 3, 2])
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1] * self.window_size[2]
        relative_position_index_flat = tf.reshape(self.relative_position_index, (-1,))
        
        #backend.numpy.take(x, indices, axis=axis)
        relative_position_bias = tf.gather(self.relative_position_bias_table,
                                           relative_position_index_flat, 
                                           axis=0,)
        
        relative_position_bias = tf.reshape(relative_position_bias,
                                            (num_window_elements, num_window_elements, -1),)
        
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), "float32")
            attn = tf.reshape(attn, (-1, nW, self.num_heads, size, size)) + mask_float
            attn = tf.reshape(attn, (-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=[0, 2, 1, 3])
        x_qkv = tf.reshape(x_qkv, (-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

# 3-D SwinTransformer block
class SwinTransformer_3d(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, 
                 num_mlp=1024, qkv_bias=True, dropout_rate=0.0, **kwargs,):
        super().__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention_3d(dim,
                                    window_size=(self.window_size, self.window_size, self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    dropout_rate=dropout_rate,)
        
        self.drop_path = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = keras.Sequential(
            [layers.Dense(num_mlp),
             layers.Activation(keras.activations.gelu),
             layers.Dropout(dropout_rate),
             layers.Dense(dim),
             layers.Dropout(dropout_rate),])

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            time, height, width = self.num_patch
            
            t_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None),)
            
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None),)
            
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None),)
            
            mask_array = np.zeros((1, time, height, width, 1))
            count = 0
            for t in t_slices:
                for h in h_slices:
                    for w in w_slices:
                        mask_array[:, t, h, w, :] = count
                        count += 1
                    
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition_3d(mask_array, self.window_size)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size * self.window_size])
            
            # copied from 2-D SwinT attention masking code 
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            
            self.attn_mask = tf.Variable(initial_value=attn_mask,
                                         shape=attn_mask.shape,
                                         dtype=attn_mask.dtype,
                                         trainable=False,)
            
    def call(self, x, training=False):
        time, height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, (-1, time, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size, -self.shift_size], axis=[1, 2, 3])
        else:
            shifted_x = x

        x_windows = window_partition_3d(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size * self.window_size, channels))
        
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = tf.reshape(attn_windows,(-1, self.window_size, self.window_size, self.window_size, channels),)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, time, height, width, channels)
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size, self.shift_size], axis=[1, 2, 3])
        else:
            x = shifted_x

        x = tf.reshape(x, (-1, time * height * width, channels))
        x = self.drop_path(x, training=training)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x


def window_partition_2d(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(x, (-1, patch_num_y, window_size, patch_num_x, window_size, channels,),)
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, (-1, window_size, window_size, channels))
    return windows


def window_reverse_2d(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(windows, (-1, patch_num_y, patch_num_x, window_size, window_size, channels,),)
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, (-1, height, width, channels))
    return x

class WindowAttention_2d(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs,):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

        num_window_elements = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)

        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.keras.initializers.Zeros(),
            name='SwinT_WindowAttention',
            trainable=True,)
        
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=relative_position_index,
            shape=relative_position_index.shape,
            dtype="int32",
            trainable=False,)

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, (-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=[0, 1, 3, 2])
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(self.relative_position_index, (-1,))
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            relative_position_index_flat,
            axis=0,)
        relative_position_bias = tf.reshape(
            relative_position_bias, (num_window_elements, num_window_elements, -1),)
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), "float32")
            attn = tf.reshape(attn, (-1, nW, self.num_heads, size, size)) + mask_float
            attn = tf.reshape(attn, (-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=[0, 2, 1, 3])
        x_qkv = tf.reshape(x_qkv, (-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

class SwinTransformer_2d(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, 
                 num_mlp=1024, qkv_bias=True, dropout_rate=0.0, **kwargs,):
        super().__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention_2d(dim,
                                       window_size=(self.window_size, self.window_size),
                                       num_heads=num_heads,
                                       qkv_bias=qkv_bias,
                                       dropout_rate=dropout_rate,)
        self.drop_path = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [layers.Dense(num_mlp),
             layers.Activation(keras.activations.gelu),
             layers.Dropout(dropout_rate),
             layers.Dense(dim),
             layers.Dropout(dropout_rate),])

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None),)
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None),)
            
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition_2d(mask_array, self.window_size)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask,
                                         shape=attn_mask.shape,
                                         dtype=attn_mask.dtype,
                                         trainable=False,)

    def call(self, x, training=False):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, (-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        x_windows = window_partition_2d(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size, channels))
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(attn_windows, (-1, self.window_size, self.window_size, channels),)
        shifted_x = window_reverse_2d(attn_windows, self.window_size, height, width, channels)
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x

        x = tf.reshape(x, (-1, height * width, channels))
        x = self.drop_path(x, training=training)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x







