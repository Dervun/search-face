import os
from typing import AnyStr, List, Tuple

import face_recognition
import numpy as np
from PIL import Image


class LinearIndex:
    def __init__(self, embeddings: List[np.ndarray] = None,
                 ids: List[int] = None):
        self.embeddings = []
        if embeddings is not None:
            self.embeddings = list(embeddings)
        if ids is None:
            self.indices2ids = {i: i for i in range(len(self.embeddings))}
        else:
            self.indices2ids = {i: ids[i] for i in range(len(self.embeddings))}

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
        ids = [self.indices2ids[i] for i in unzipped[1]]
        return distances, ids

    def add_new_embedding(self, new_embedding: np.ndarray, new_id: int):
        if self.embeddings and new_embedding.shape != self.embeddings[0].shape:
            raise ValueError(f'Expected {self.embeddings[0].shape}, '
                             f'got {new_embedding.shape}')
        self.embeddings.append(new_embedding)
        self.indices2ids[len(self.embeddings) - 1] = new_id

    def get_all_embeddings(self):
        return self.embeddings


class BigBrother:
    def __init__(self, embeddings=None, names=None, top_k: int = 5,
                 model: str = 'small', data_path: AnyStr = None):
        self.index = LinearIndex(embeddings)

        self.names = {}
        if names is not None:
            self.names = dict(names)

        self.top_k = top_k
        if model not in ['small', 'large']:
            raise ValueError(f'Expected `model` is one of "small" or "large"')
        self.model = model

        self.id_counter = 0
        if names:
            self.id_counter = max(self.names.keys()) + 1

        if data_path is None:
            data_path = '.'
        images = os.path.join(data_path, 'images')
        if not os.path.isdir(images):
            os.makedirs(images)
        self.images = images

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
        distances, ids = self.index.get_nearest(face_encoding, top_k)
        names = [self.names[i] for i in ids]
        return distances, ids, names

    def add_new_face(self, image: np.ndarray, name: str):
        # TODO: add check for duplicate
        face_locations = [BigBrother._get_biggest_face_location(image)]
        face_encoding = face_recognition.face_encodings(
            image, face_locations, model=self.model)[0]
        self.names[self.id_counter] = name
        self.index.add_new_embedding(face_encoding, self.id_counter)
        pil_image = Image.fromarray(image)
        pil_image.save(os.path.join(self.images, f'{self.id_counter}.png'),
                       'PNG')
        self.id_counter += 1

    def get_all_data(self):
        return {'embeddings': self.index.get_all_embeddings(),
                'names': self.names, 'top_k': self.top_k, 'model': self.model,
                'data_path': self.images}

    def get_images_folder(self) -> AnyStr:
        return self.images

