from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import cv2
from modeling_clipseg import CLIPSegForImageSegmentation
from processing_clipseg import CLIPSegProcessor

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

image = np.array(Image.open("test4.png"))[:,:,:-1]
image = cv2.resize(image, (512, 384), interpolation=cv2.INTER_AREA)
H,W,C = image.shape


texts=["pavement","car","building","pool", "crosswalk", "truck", "boat", "sidewalk", "tree", "vegetation"]
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    start = time.perf_counter()
    outputs = model(**inputs)
    end = time.perf_counter()
 
    elapsed = end - start
    print(f"Model Inference Time: {elapsed:.6f} sec")
    print("len: ", len(outputs))
