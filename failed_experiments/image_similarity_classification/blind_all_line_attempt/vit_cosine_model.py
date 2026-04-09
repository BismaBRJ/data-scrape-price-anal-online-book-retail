"""
This file defines the overall structure of my
image similarity classification model:
a ViT (Vision Transformer) for vector embeddings/encoding,
followed by the calculation of cosine similarity.
Later the threshold is determined with
either the Precision-Recall curve or linear SVM's decision boundary.

Possible error I know of:
Something something SSL certificate, preceded by something like
Downloading: "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth"
to [some directory]
Solution: manually download that file and move it to that directory
(funny to think PyTorch of all organizations can have expired SSL certificates)
"""

# Imports
import torch
import torchvision.transforms as transforms
#from torchvision.models import list_models
#print(list_models())
from torchvision.models import vit_b_32, ViT_B_32_Weights

# Code

def trim_base64_prefix(x):
    if ((len(x) >= 23) and (x[:23] == "data:image/jpeg;base64,")):
        return x[23:]
    else:
        return x

# function to make image suitable for encoding
img_to_encodable = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

# multiple encoders can be instantiated (with independent weights),
# for whatever reason
def MyEncoder():
    return vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)

# however there is no need for multiple cosine similarity functions
my_cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

# lambda functions could be cumbersome in this case, so...
def my_pred_sim(*args, **kwargs): # really just maps [-1, 1] -> [0, 1]
    return (my_cos_sim(*args, **kwargs) + 1)/2

# general outline, unused?
class EncodeCosine(torch.nn.Module):
    def __init__(self):
        #self.THRESHOLD = SIMILARITY_THRESHOLD
        self.encode = MyEncoder()
        self.cos_sim = my_cos_sim
    
    def forward(self, x):
        # inputted x: torch.stack([img1, img2], dim=0)
        # dim of img1, img2 each is (3, 224, 224)
        # thus dim of x is (2, 3, 224, 224)
        vecs = self.encode(x)
        vec1 = vecs[0]
        vec2 = vecs[1]
        cos_sim_score = self.cos_sim(vec1, vec2) # [-1, 1]
        sim_score = (cos_sim_score + 1)/2 # [0, 1]
        #is_similar = sim_score >= self.THRESHOLD
        return sim_score
