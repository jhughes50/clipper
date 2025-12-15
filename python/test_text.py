import torch
from processing_clipseg import CLIPSegProcessor
import numpy as np
import cv2
from PIL import Image

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = torch.jit.load("saved-models/clip-vision-model-traced.pt")


image = np.array(Image.open("test4.png"))[:,:,:-1]
image = cv2.resize(image, (512, 384), interpolation=cv2.INTER_AREA)

inputs = processor(text=["parking lot"], images=image, return_tensors='pt',padding="max_length",max_length=10)

print("ids: ", inputs["input_ids"])
print("ids: ", type(inputs["input_ids"]))
print("ids: ", len(inputs["input_ids"][0]))
print("am: ", inputs["attention_mask"])
output = model(inputs["pixel_values"])

print(type(output))
print("inner tuple types: ", type(output[2][0]))
for e in output:
    print(type(e))
