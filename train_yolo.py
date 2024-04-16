from ultralytics import YOLO
from PIL import Image
import numpy as np
# Load a model



# 
model = YOLO('/nas_data/WTY/project/nlp_task1/runs/detect/train2/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
# model = YOLO('/nas_data/WTY/project/nlp_task1/runs/detect/train2/weights/best.pt')
# results = model.train(data='custom.yaml', epochs=100, imgsz=640, device=[1,2])

results = model.predict(['1000_0003.jpg'], conf=0.5)


# sort crop images
def regularization_sorting(Lx, Ly, W, H, alpha, beta):
    sorted_Lx, sorted_Ly, sorted_W, sorted_H = [], [], [], []

    while len(Lx):
        # Step 3: Calculate the average value M within the range of alpha for the minimum values of Ly
        min_Ly = min(Ly)
        M = sum(y for y in Ly if y <= min_Ly + alpha) / len([y for y in Ly if y <= min_Ly + alpha])

        # Step 4: Treat the index i of characters that are within a distance of beta from the mean M
        indices = [i for i, y in enumerate(Ly) if abs(y - M) <= beta]

        # Step 5: Sort i according to horizontal coordinate from small to large
        indices.sort(key=lambda i: Lx[i])

        # Step 6: Take sorted coordinates according to i into sorted_Lx, sorted_Ly, sorted_W, sorted_H
        for i in indices:
            sorted_Lx.append(Lx[i])
            sorted_Ly.append(Ly[i])
            sorted_W.append(W[i])
            sorted_H.append(H[i])

        # Step 7: Remove the coordinates already taken from Lx, Ly, W, H
        Lx = [x for i, x in enumerate(Lx) if i not in indices]
        Ly = [y for i, y in enumerate(Ly) if i not in indices]
        W = [w for i, w in enumerate(W) if i not in indices]
        H = [h for i, h in enumerate(H) if i not in indices]

    return sorted_Lx, sorted_Ly, sorted_W, sorted_H


# convert x y w h n to lx ly w h
def convert_boxes(boxes, size):
    for box in boxes:
        box[0] = box[0] * size[0]
        box[1] = box[1] * size[1]
        box[2] = box[2] * size[0]
        box[3] = box[3] * size[1]
        box[0] = box[0] - box[2] / 2  
        box[1] = box[1] - box[3] / 2

    return boxes  


# eg. crop and sort images
for r in results:
    img = Image.open('1000_0003.jpg')
    size = img.size
    boxes = r.boxes.xywhn.cpu().numpy()
    boxes = convert_boxes(boxes, size)

    sb1,sb2,sb3, sb4 = regularization_sorting(boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], alpha=45, beta=30)
    sb1,sb2,sb3, sb4 = np.expand_dims(sb1, axis=1), np.expand_dims(sb2, axis=1), np.expand_dims(sb3, axis=1), np.expand_dims(sb4, axis=1)
    boxes = np.concatenate([sb1, sb2, sb3, sb4], axis=1)

    for i, box in enumerate(boxes):
        x, y, w, h = box
        print(x, y, w, h)
        crop = img.crop((x, y, x+w, y+h))
        crop.save(f'results/{i}.jpg')

    # from PIL import ImageDraw

    # draw = ImageDraw.Draw(img)
    # for box in boxes:
    #     x, y, w, h = box
    #     draw.rectangle([x, y, x+w, y+h], outline='red')
    #     img.save('result.jpg')
    # print(boxes)

# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     # result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk