from typing import List, Tuple

import face_recognition
import numpy as np


class LinearIndex:
    def __init__(self, embeddings: List[np.ndarray] = None):
        self.embeddings = []
        if embeddings is not None:
            self.embeddings = list(embeddings)

    def __len__(self):
        return len(self.embeddings)

    def get_nearest(self, embedding: np.ndarray, top_k: int) \
            -> Tuple[List[float], List[int]]:
        face_distances = face_recognition.face_distance(
            self.embeddings, embedding)
        top_k = min(top_k, len(self.embeddings))
        top_indices = np.argpartition(face_distances, top_k - 1)[:top_k]
        unzipped = tuple(zip(
            *sorted([(face_distances[i], int(i)) for i in top_indices])))
        distances = list(unzipped[0])
        indices = list(unzipped[1])
        return distances, indices

    def add_new_embedding(self, new_embedding):
        if self.embeddings and new_embedding.shape != self.embeddings[0].shape:
            raise ValueError(f'Expected {self.embeddings[0].shape}, '
                             f'got {new_embedding.shape}')
        self.embeddings.append(new_embedding)

    def get_all_embeddings(self):
        return self.embeddings


class BigBrother:
    def __init__(self, embeddings=None, names=None, top_k: int = 5,
                 model: str = 'small'):
        self.index = LinearIndex(embeddings)
        self.names = []
        if names is not None:
            self.names = list(names)
        self.top_k = top_k
        if model not in ['small', 'large']:
            raise ValueError(f'Expected `model` is one of "small" or "large"')
        self.model = model

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _get_biggest_face_location(image: np.ndarray) \
            -> Tuple[int, int, int, int]:
        face_locations = face_recognition.face_locations(image)
        sizes = [(bottom - top) * (right - left)
                 for top, right, bottom, left in face_locations]
        biggest_index = np.argpartition(np.array(sizes), len(sizes) - 1)[-1]
        face_location = face_locations[biggest_index]
        return face_location

    def get_nearest(self, image: np.ndarray, top_k: int = None):
        """image is NumPy ndarray, shape is height, width, channels (3)
        values from 0 to 255
        """
        # process only the biggest face
        face_locations = [BigBrother._get_biggest_face_location(image)]
        face_encoding = face_recognition.face_encodings(
            image, face_locations, model=self.model)[0]
        if top_k is None:
            top_k = self.top_k
        distances, indices = self.index.get_nearest(face_encoding, top_k)
        names = [self.names[i] for i in indices]
        return distances, indices, names

    def add_new_face(self, image: np.ndarray, name):
        # TODO: add check for duplicate
        face_locations = [BigBrother._get_biggest_face_location(image)]
        face_encoding = face_recognition.face_encodings(
            image, face_locations, model=self.model)[0]
        self.names.append(name)
        self.index.add_new_embedding(face_encoding)

    def get_all_data(self):
        return {'embeddings': self.index.get_all_embeddings(),
                'names': self.names, 'top_k': self.top_k, 'model': self.model}

