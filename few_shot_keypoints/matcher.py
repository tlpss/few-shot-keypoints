from typing import List, Tuple

import numpy as np
import torch


class KeypointFeatureMatcher:
    """
    A class to find best matching (u,v) coordinates for an image 
    against a set of (image, keypoint) reference pairs using a dense feature extractor.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.reference_images = []
        self.reference_image_keypoints = []
        self.reference_vectors = []


    def add_reference_image(self, image: np.ndarray, keypoint: List[int]):
        assert image.shape[0] == 1
        assert len(image.shape) == 4 # 1,C,H,W
        vector = self.feature_extractor.extract_features(image)[:,:,keypoint[1],keypoint[0]]
        self.reference_images.append(image)
        self.reference_image_keypoints.append(keypoint)
        self.reference_vectors.append(vector)


    def get_best_match(self, img: np.ndarray) -> Tuple[int,int]:
        assert len(img.shape) == 4 # 1,C,H,W
        assert img.shape[0] == 1
        image_features = self.feature_extractor.extract_features(img) # 1,D,H,W
        image_features = image_features[0] # D,H,W
        
        reference_vector = self.reference_vectors[0] #1,D
        reference_vector = reference_vector.unsqueeze(2).unsqueeze(3) # 1,D,1,1
        cos_map = torch.nn.functional.cosine_similarity(image_features, reference_vector,dim=1) # 1,H,W

        argmax = cos_map.argmax()
        _,v,u = torch.unravel_index(argmax, cos_map.shape)
        return (u.item(),v.item())





class MultiQueryKeypointFeatureMatcher(KeypointFeatureMatcher):
    def get_best_match(self, img: np.ndarray) -> List[Tuple[int,int]]:
        """ iterate over all reference_vectors, for each reference vector find the best match.
        Then, for each match, calculate the similarity score for all query keypoints and sum them up. 
        return the best match for each query keypoint"""
        assert len(img.shape) == 4 # 1,C,H,W
        assert img.shape[0] == 1
        image_features = self.feature_extractor.extract_features(img) # 1,D,H,W
        image_features = image_features[0] # D,H,W
        
        best_matches = []
        for reference_vector in self.reference_vectors:
            reference_vector = reference_vector.unsqueeze(2).unsqueeze(3) # 1,D,1,1
            cos_map = torch.nn.functional.cosine_similarity(image_features, reference_vector,dim=1) # 1,H,W
            argmax = cos_map.argmax()
            _,v,u = torch.unravel_index(argmax, cos_map.shape)
            best_matches.append((u.item(),v.item(),image_features[:,v,u]))

        # select the best match using the sum of similarities
        total_similarity_scores = {}
        for candidate in best_matches:
            similarity_score = 0
            candidate_u, candidate_v, candidate_vector = candidate
            candidate_vector = candidate_vector.unsqueeze(0)
            for reference_vector in self.reference_vectors:
                score = torch.nn.functional.cosine_similarity(candidate_vector, reference_vector,dim=1)
                similarity_score += score.item()
            total_similarity_scores[candidate] = similarity_score

        # select the best match using the sum of similarities
        best_match = sorted(total_similarity_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        return (best_match[0], best_match[1])