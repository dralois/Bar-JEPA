import requests
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor

torch.set_default_device("cuda")

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"
url_3 = "https://farm4.staticflickr.com/3174/2588589960_23b82d1114_z.jpg"
url_4 = "https://farm3.staticflickr.com/2762/4398530289_d770d5da01_z.jpg"

image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)
image_3 = Image.open(requests.get(url_3, stream=True).raw)
image_4 = Image.open(requests.get(url_4, stream=True).raw)

model_id = "facebook/ijepa_vith14_1k"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)


def infer(image):
    inputs = processor(image, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


embed_1 = infer(image_1)
embed_2 = infer(image_2)
embed_3 = infer(image_3)
embed_4 = infer(image_4)

similarity = cosine_similarity(embed_1, embed_2)
print(similarity)