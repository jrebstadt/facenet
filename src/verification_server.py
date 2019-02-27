from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import shuffle
from scipy import misc
import os
from external_call_simulation.authenticatee import Authenticatee
from src import embedding_computation

PATH_RANDOM_PICTURE_1 = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), "data/images/Aaron_Peirsol_0003.png"))
PATH_RANDOM_PICTURE_2 = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), "data/images/Abel_Aguilar_0001.png"))
PATH_RANDOM_PICTURE_3 = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), "data/images/Al_Davis_0002.png"))
PATH_RANDOM_PICTURE_4 = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), "data/images/Alyse_Beaupre_0001.png"))



class VerificationServer:

    def __init__(self):
        self.pathes_for_cross_checking = [PATH_RANDOM_PICTURE_1, PATH_RANDOM_PICTURE_2, PATH_RANDOM_PICTURE_3, PATH_RANDOM_PICTURE_4]
        self.embeddings_for_cross_checking = self.read_images(self.pathes_for_cross_checking)

    def read_images(self, pathes_for_cross_checking):
        authenticatee = Authenticatee()
        images_for_cross_checking = [misc.imread(os.path.expanduser(image_path), mode='RGB') for image_path in pathes_for_cross_checking]
        embeddings_for_cross_checking = [embedding_computation.compute_embedding(image) for image in images_for_cross_checking]
        embeddings_for_cross_checking = [(False, embedding, authenticatee.compute_distance(embedding)) for embedding in embeddings_for_cross_checking]
        return embeddings_for_cross_checking

    def compute_score(self, image):

        authenticatee = Authenticatee()
        embedding_of_correct_image = embedding_computation.compute_embedding(image)
        embeddings_to_verify = self.embeddings_for_cross_checking
        embeddings_to_verify.append((True, embedding_of_correct_image, 0))
        shuffle(embeddings_to_verify)
        score = None
        delta = 0.02
        for embedding_tupel in embeddings_to_verify:
            correct_value, embedding, distance = embedding_tupel
            if correct_value:
                score = authenticatee.compute_distance(embedding)
                print('new score computed (' + str(score) + ')')
            else:
                distance_computed_by_authenticatee = authenticatee.compute_distance(embedding)
                difference = distance_computed_by_authenticatee - distance
                absolute_difference_substracted_by_delta = abs(difference) - delta
                print('absolute difference: ' + str(absolute_difference_substracted_by_delta))
                if absolute_difference_substracted_by_delta > 0:
                    return None
        print('returning score: ' + str(score))
        return score

