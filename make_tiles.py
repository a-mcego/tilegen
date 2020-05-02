import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

#generates a tilemap from a big single PNG file

def extract_tiles(filename, tilesize):
    pic = tf.io.decode_png(tf.io.read_file(filename))
    picsize = [pic.shape[0],pic.shape[1]]
    pic = tf.reshape(pic,[picsize[0]//tilesize[0],tilesize[0],picsize[1]//tilesize[1],tilesize[1],3])
    pic = tf.transpose(pic,[0,2,1,3,4])
    pic = tf.reshape(pic,[picsize[0]//tilesize[0]*picsize[1]//tilesize[1],tilesize[0],tilesize[1],3])
    pic = pic.numpy()
    tiles, indices = np.unique(pic,axis=0,return_inverse=True)
    indices = np.reshape(indices,[picsize[0]//tilesize[0],picsize[1]//tilesize[1]])
    print(tilesize, tiles.shape[0], tiles.shape[0]/indices.shape[0])
    print(tiles.shape, indices.shape)

    np.savez_compressed(f"{filename}.npz",tiles=tiles,indices=indices)

extract_tiles("pokemon_firered_leafgreen.png", [16,16])