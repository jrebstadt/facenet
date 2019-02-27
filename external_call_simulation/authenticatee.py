from src import embedding_computation
from scipy import misc
import numpy as np
import os


class Authenticatee:

    def __init__(self):
        self.image_path = '/media/jonas/data/git_repositories/facenet/data/images/Anthony_Hopkins_0002.jpg'
        self.image = misc.imread(os.path.expanduser(self.image_path), mode='RGB')
        self.own_embedding = embedding_computation.compute_embedding(self.image)

    def compute_distance(self, embedding_authenticator):

        distance = np.sqrt(np.sum(np.square(np.subtract(embedding_authenticator[:], self.own_embedding[:]))))

        return distance