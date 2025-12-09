import torch
from processing_clipseg import CLIPSegProcessor
import numpy as np
import cv2
from PIL import Image

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
text_model = torch.jit.load("clip-text-model-traced.pt")


image = np.array(Image.open("test4.png"))[:,:,:-1]
image = cv2.resize(image, (512, 384), interpolation=cv2.INTER_AREA)

inputs = processor(text=["parking lot"], images=image, return_tensors='pt',padding="max_length",max_length=10)

print("ids: ", inputs["input_ids"])
print("ids: ", type(inputs["input_ids"]))
print("ids: ", len(inputs["input_ids"][0]))
print("am: ", inputs["attention_mask"])
output = text_model(inputs["input_ids"], inputs["attention_mask"])
