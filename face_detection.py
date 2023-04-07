import os
import pickle
from typing import Tuple, List

import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


# Constans Types
IMAGE_TYPE = Image.Image
NUMPY_ARRAY_TYPE = np.ndarray[(1,512)] 
REFERENCE_EMBEDDINGS_TYPE = List[Tuple[str, NUMPY_ARRAY_TYPE]] 

class FaceDetector:

    
    def __init__(self, embeddings_path: str = None, decision_limit: float = 1.0):
        """
        This is the constructor of the class.

        arguments:
            -   reference_embeddings: Dict[str, numpy_array(1, 512)]
            (a list of tuples, where each tuple contains in the first position 
            the id of the embedding and in the second the embedding).
        """
        # Modelo MTCNN para detección y recorte de rostros
        self.mtcnn = MTCNN()
        # Modelo InceptionResnetV1 pre-entrenado para reconocimiento de caras
        self.inception_resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.decision_limit = decision_limit
        self.load_reference_embeddings(path=embeddings_path)

    def set_reference_embeddings(self, reference_embeddings: REFERENCE_EMBEDDINGS_TYPE) -> None:
        self.reference_embeddings = reference_embeddings

    def get_reference_embeddings(self) -> REFERENCE_EMBEDDINGS_TYPE:
        return self.reference_embeddings
    
    def set_decision_limit(self, decision_limit: int) -> None:
        self.decision_limit = decision_limit

    def get_decision_limit(self) -> int:
        return self.decision_limit
    
    def add_embedding_to_reference(self, embedding: Tuple[str, NUMPY_ARRAY_TYPE]) -> None:
        """Add a new embedding/id pair to reference_embeddings."""
        self.reference_embeddings.append(embedding)
    
    def get_embedding_by_image(self, image: IMAGE_TYPE) -> NUMPY_ARRAY_TYPE:
        """
        The embedding of the image passed by parameter are calculated using MTCNN and InceptionResnetV1.
        if no face is recognized, a numpy(1,512) of ones is returned.
        We choose this vector to represent the unknown embedding because it is away from any 
        embedding that may be returned by faceNet.
        """
        # The face of the image is detected and cropped
        faces, _ = self.mtcnn(image, return_prob=True)

        if faces is None:
            return np.ones((1, 512))

        # Embedding of each face is calculated using InceptionResnetV1
        embeddings = self.inception_resnet(faces.reshape(1,faces.shape[0],faces.shape[1],faces.shape[2]))

        return embeddings.detach().numpy()
    
    def _get_id_by_euclidean_distance(self, unknown_embedding: NUMPY_ARRAY_TYPE) -> str:
        """
        This function retrieves the id of unknown_encoding and confidence. To do this, 
        the Euclidean distance of unknown_encoding is compared against the rest of the 
        embeddings contained in self.reference_embeddings, keeping the emsbedding and 
        its id with the smallest Euclidean distance.
        The condifence is the smallest Euclidean distance, that is, the closer to zero, the greater the confidence.
        """
        euclidan_distances = map(
            lambda embedding_tuple: (embedding_tuple[0], np.linalg.norm(unknown_embedding - embedding_tuple[1])),
            self.reference_embeddings
        )
        embedding_id, euclidan_distance = min(euclidan_distances, key=lambda x: x[1])

        return (embedding_id, euclidan_distance) if euclidan_distance < self.decision_limit else ('stranger', 0)
    
    def get_id_by_image(self, image: IMAGE_TYPE) -> str:
        """
        This function will return the id belonging to the person in the image
        and the confidence, if the embeddings that representing that person is stored 
        in self.reference_embeddings.
        """
        return self._get_id_by_euclidean_distance(self.get_embedding_by_image(image))
    
    def save_reference_embeddings(self, path: str = None) -> None:
        """
        This function stores reference_embeddings in the path passed as an argument. If no path is specified, 
        the reference_embeddings are saved in ./enbeddings/reference_enbeddings.
        """
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__), 
                'embeddings/reference_embeddings.pkl'
            )

        with open(path, 'wb') as file:
            pickle.dump(self.reference_embeddings, file)

        file.close()

    def load_reference_embeddings(self, path: str = None) -> None:
        """This function loads the reference_embeddings that are specified in the path argument."""
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__), 
                'embeddings/reference_embeddings.pkl'
            )

        with open(path, 'rb') as file:
             self.reference_embeddings = pickle.load(file)

        file.close()
        

    def calculate_reference_embeddings(self, images_with_id: List[Tuple[str, IMAGE_TYPE]]) -> None:
        """
        This function calculates the embedding of each image and saves them together with their 
        id in self.reference_embeddings.

        args:
            - images_with_id: List of tuples, where each tuple contains an image and its associated id
        """
        self.reference_embeddings = list(map(
            lambda image_tuple: (image_tuple[0], self.get_embedding_by_image(image_tuple[1])),
            images_with_id
        )) 