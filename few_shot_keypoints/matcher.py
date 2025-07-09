from typing import Any, List, Tuple
import abc
import numpy as np
import torch

from few_shot_keypoints.featurizers.base import BaseFeaturizer
from dataclasses import dataclass

@dataclass
class MatchingResult:
    #TODO: what is the expected behavior for keypoints that are not visible?
    u: int
    v: int
    score: float

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)



class BaseKeypointFeatureMatcher(abc.ABC):
    def __init__(self):
        pass 

    @abc.abstractmethod
    def add_reference_image(self, image: torch.Tensor, keypoint: List[int]):
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def get_best_match(self, img: torch.Tensor) -> MatchingResult:
        raise NotImplementedError("Subclasses must implement this method")

    def validate_input(self, img: torch.Tensor):
        assert len(img.shape) == 4 # 1,C,H,W
        assert img.shape[0] == 1
        # check in range (0,1)
        assert img.min() >= 0 and img.max() <= 1, "Image must be in range (0,1)"

class KeypointFeatureMatcher(BaseKeypointFeatureMatcher):
    """
    A class to find best matching (u,v) coordinates for an image 
    against a set of (image, keypoint) reference pairs using a  feature extractor.
    """
    def __init__(self, feature_extractor: BaseFeaturizer):
        self.feature_extractor = feature_extractor
        self.reference_images = []
        self.reference_image_keypoints = []
        self.reference_vectors = []


    def add_reference_image(self, image: torch.Tensor, keypoint: List[int]):
        self.validate_input(image)
        keypoint = [round(k) for k in keypoint]
        assert len(keypoint) == 2, "Keypoint must be a list of two integers"
        vector = self.feature_extractor.extract_features(image)[:,:,keypoint[1],keypoint[0]]
        self.reference_images.append(image)
        self.reference_image_keypoints.append(keypoint)
        self.reference_vectors.append(vector)


    def get_best_match(self, img: torch.Tensor) -> MatchingResult:
        self.validate_input(img)
        image_features = self.feature_extractor.extract_features(img) # 1,D,H,W
        image_features = image_features[0] # D,H,W
        
        reference_vector = self.reference_vectors[0] #1,D
        reference_vector = reference_vector.unsqueeze(2).unsqueeze(3) # 1,D,1,1
        cos_map = torch.nn.functional.cosine_similarity(image_features, reference_vector,dim=1) # 1,H,W

        argmax = cos_map.argmax()
        _,v,u = torch.unravel_index(argmax, cos_map.shape)
        return MatchingResult(u=int(u.item()), v=int(v.item()), score=round(float(cos_map[0,v,u].item()), 4))





class MultiQueryKeypointFeatureMatcher(KeypointFeatureMatcher):
    def get_best_match(self, img: torch.Tensor) -> MatchingResult:
        """ iterate over all reference_vectors, for each reference vector find the best match.
        Then, for each match, calculate the similarity score for all query keypoints and sum them up. 
        return the best match for each query keypoint"""
        self.validate_input(img)
        image_features = self.feature_extractor.extract_features(img) # 1,D,H,W
        image_features = image_features[0] # D,H,W
        
        match_candidates = []
        for reference_vector in self.reference_vectors:
            reference_vector = reference_vector.unsqueeze(2).unsqueeze(3) # 1,D,1,1
            cos_map = torch.nn.functional.cosine_similarity(image_features, reference_vector,dim=1) # 1,H,W
            argmax = cos_map.argmax()
            _,v,u = torch.unravel_index(argmax, cos_map.shape)
            match_candidates.append((u.item(),v.item(),image_features[:,v,u]))

        # select the best match using the sum of similarities
        total_similarity_scores = {}
        for candidate in match_candidates:
            similarity_score = 0
            candidate_u, candidate_v, candidate_vector = candidate
            candidate_vector = candidate_vector.unsqueeze(0)
            for reference_vector in self.reference_vectors:
                score = torch.nn.functional.cosine_similarity(candidate_vector, reference_vector,dim=1)
                similarity_score += score.item()
            total_similarity_scores[candidate] = similarity_score

        # select the best match using the sum of similarities
        best_match = sorted(total_similarity_scores.items(), key=lambda x: x[1], reverse=True)[0]
        best_candidate, best_score = best_match
        return MatchingResult(u=int(best_candidate[0]), v=int(best_candidate[1]), score=round(best_score, 4))



class KeypointListMatcher:
    """
    A class to match keypoints in an image to a set of reference images.
    """
    def __init__(self, keypoint_channels: List[str], matchers: List[BaseKeypointFeatureMatcher]):
        self.keypoint_channels = keypoint_channels
        self.matchers = matchers
        assert len(self.keypoint_channels) == len(self.matchers), f"number of keypoint channels ({len(self.keypoint_channels)}) must match number of matchers ({len(self.matchers)})"

    def get_keypoints(self, img: torch.Tensor) -> List[MatchingResult]:
        return [matcher.get_best_match(img) for matcher in self.matchers]


