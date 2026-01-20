from dataclasses import dataclass
import time
from typing import Optional, List

import torch

@dataclass
class MatchingResult:
    u: int
    v: int
    score: float

def custom_cos_sim(image_features: torch.Tensor, reference_vectors: torch.Tensor) -> torch.Tensor:
    """
    image_features: (1,D,H,W)
    reference_vectors: (N,D)
    returns a tensor of shape (N,H,W)
    """
    # reference_vectors = reference_vectors.unsqueeze(2).unsqueeze(3) # (N,D,1,1)
    # return torch.nn.functional.cosine_similarity(image_features, reference_vectors, dim=1)
    # clone both 
    normalized_image_features = image_features / torch.clamp(image_features.norm(dim=1, keepdim=True,p=2), min=1e-8)
    normalized_reference_vectors = reference_vectors / torch.clamp(reference_vectors.norm(dim=1, keepdim=True,p=2), min=1e-8)
    matrix = torch.matmul(normalized_image_features.permute(0,2,3,1), normalized_reference_vectors.transpose(0,1))
    permuted_matrix = matrix.permute(0,3,1,2)
    return permuted_matrix[0]

class KeypointFeatureMatcher:
    """
    Find best matching (u,v) coordinates for image features 
    against a reference feature vector.
    
    Used for correspondence matching.
    """
    def __init__(self, reference_vectors: torch.Tensor, top_k_matches: List[int] = None, min_distance_between_topk_matches: int = 50, device: str = "cuda:0"):
        # D dimensional vectors.
        self.reference_vectors = reference_vectors.to(device)
        assert len(self.reference_vectors.shape) == 2, "Reference vectors must be a 2D tensor"
        if top_k_matches is None:
            self.top_k_matches = [1] * len(reference_vectors)
        else:
            self.top_k_matches = top_k_matches
            assert len(self.top_k_matches) == len(reference_vectors), "Top k matches must be the same length as reference vectors"
        self.min_distance_between_topk_matches = min_distance_between_topk_matches
        self.device = torch.device(device)


    # def get_best_matches_from_image(self, image: torch.Tensor, mask:Optional[torch.Tensor] = None) -> List[List[MatchingResult]]:
    #     image_features = self.keypoint_featurizer.extract_features(image).to(self.device)
    #     return self.get_best_matches_from_image_features(image_features, mask)

    def get_best_matches_from_image_features(self, image_features: torch.Tensor, mask:Optional[torch.Tensor] = None) -> List[List[MatchingResult]]:
        self.validate_image_features_input(image_features)
        reference_vectors = self.reference_vectors.unsqueeze(2).unsqueeze(3) # N,D,1,1
        # cosine similarity handles the normalization internally.
        # cos_map = torch.nn.functional.cosine_similarity(image_features, reference_vectors,dim=1)# N,H,W
        # manually compute the cosine similarity to make it a lot faster.. no clue why this is? 
        cos_map = custom_cos_sim(image_features, reference_vectors.squeeze(2).squeeze(2))

        return self.get_best_matches_from_similarities(cos_map, mask)


    def get_best_matches_from_similarities(self, cosine_similarities: torch.Tensor,mask:Optional[torch.Tensor] = None) -> List[List[MatchingResult]]:
        """
        cosine_similarities: N,H,W
        """

        self.validate_cosine_similarities_input(cosine_similarities)
        cos_map = cosine_similarities.clone() # make a copy to avoid modifying the original map
        if mask is not None:
            self.validate_mask(mask)
            assert cosine_similarities.shape[-2:] == mask.shape[-2:], "Mask must have the same shape as img"
            # only consider matches that are within the mask
            #cos_map = cos_map * mask.unsqueeze(0).to(cos_map.device)
            cos_map[:,mask==False] = -1

        results = []
        for ref_idx in range(len(self.reference_vectors)):
            ref_cos_map = cos_map[ref_idx]
            ref_results = []
            for i in range(self.top_k_matches[ref_idx]):
                argmax = ref_cos_map.argmax()
                v,u = torch.unravel_index(argmax, ref_cos_map.shape)
                ref_results.append(MatchingResult(u=int(u.item()), v=int(v.item()), score=round(float(ref_cos_map[v,u].item()), 4)))
                # set square around the keypoint to zero, so that next best match is not in that neighborhood. (otherwise they would all be at adjacent pixels..)
                # TODO: make this a circle?
                if i < self.top_k_matches[ref_idx] - 1:
                    padding = self.min_distance_between_topk_matches
                    #TODO: limit to square within image bounds
                    ref_cos_map[v-padding:v+padding,u-padding:u+padding] = -1 
            results.append(ref_results)
            
            
        return results

    @staticmethod
    def custom_cos_sim(image_features: torch.Tensor, reference_vectors: torch.Tensor) -> torch.Tensor:
        """
        this proved to be faster than the default cosine similarity function in torch.nn.functional.cosine_similarity
        for some reason? I might be doing something wrong here..
        """
        normalized_image_features = image_features / image_features.norm(dim=1, keepdim=True,p=2)
        normalized_reference_vectors = reference_vectors / reference_vectors.norm(dim=1, keepdim=True,p=2)
        # epsilon to avoid division by zero
        epsilon = 1e-8
        normalized_image_features = torch.where(normalized_image_features < epsilon, epsilon, normalized_image_features)
        normalized_reference_vectors = torch.where(normalized_reference_vectors < epsilon, epsilon, normalized_reference_vectors)
        matrix = torch.matmul(normalized_image_features.permute(0,2,3,1), normalized_reference_vectors.transpose(0,1))
        permuted_matrix = matrix.permute(0,3,1,2)
        return permuted_matrix[0]

    def validate_image_features_input(self, image_features: torch.Tensor):
        assert len(image_features.shape) == 4 # 1,C,H,W
        assert image_features.shape[0] == 1
        # check in range (0,1)
        assert image_features.device == self.device, f"Image must be on the same device as the matcher, but got {image_features.device} and {self.device}"

    def validate_cosine_similarities_input(self, cosine_similarities: torch.Tensor):
        assert len(cosine_similarities.shape) == 3 # N,H,W
        assert cosine_similarities.shape[0] == len(self.reference_vectors), f"Cosine similarities must have the same number of reference vectors as the matcher, but got {cosine_similarities.shape[0]} and {len(self.reference_vectors)}"
        assert cosine_similarities.device == self.device, f"Cosine similarities must be on the same device as the matcher, but got {cosine_similarities.device} and {self.device}"

    def validate_mask(self, mask: torch.Tensor):
        assert len(mask.shape) == 2 # H,W
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"
        assert mask.device == self.device, f"Mask must be on the same device as the matcher, but got {mask.device} and {self.device}"


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    n_vectors = 1
    reference_vectors = torch.randint(0, 255, (n_vectors, 1000)) /255.0   
    reference_vectors = reference_vectors.to("cuda:1")
    matcher = KeypointFeatureMatcher(reference_vectors, top_k_matches=[2] * n_vectors, min_distance_between_topk_matches=50, device="cuda:1")
    image_features = torch.randint(0, 255, (1, 1000, 256, 256)) /255.0
    image_features = image_features.to("cuda:1")
    mask = torch.ones(256, 256) == 1
    mask = mask.to("cuda:1")
    results = matcher.get_best_matches_from_image_features(image_features, mask=mask)

    cosine_similarities = matcher.custom_cos_sim(image_features, reference_vectors)
    results = matcher.get_best_matches_from_similarities(cosine_similarities, mask=mask)


    image = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)
    image = Image.fromarray(image)
    print(results)
    # image = visualize_keypoints(image, results)


    # profile the python code using torch.profiler
    # with torch.profiler.profile(record_shapes=True) as prof:
    #     for i in range(1):
    #         results = matcher.get_best_matches_from_similarities(cosine_similarities, mask=mask)
    #         torch.cuda.synchronize()
    # print(prof.key_averages().table("cpu_time_total", row_limit=20))
    # # dump to file 
    # prof.export_chrome_trace("profiler_trace.json")

