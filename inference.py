import numpy as np
import cv2

# Resize images to fit model input and reduce computation time
def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image

# Convert the image from raw to cartoon using the cartoon GAN model
def cartoonize(img, model):
    batch_image = img.astype(np.float32)/127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    ort_inputs = {model.get_inputs()[0].name: batch_image}
    ort_outs = model.run(None, ort_inputs)
    img_out = ort_outs[0]
    output = (np.squeeze(img_out)+1)*127.5
    cartoon = np.clip(output, 0, 255).astype(np.uint8)
    return cartoon

# Normalize image to fit deeplabv3 input
def normalize(img):
    MEAN = 255 * np.array([0.485, 0.456, 0.406])
    STD = 255 * np.array([0.229, 0.224, 0.225])
    img = img.transpose(-1, 0, 1)
    img = (img - MEAN[:, None, None]) / STD[:, None, None]
    output = np.array([img.astype('float32')])
    return output

# Use deeplabv3 model to find person segmentation
def findmask_deeplabv3(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = normalize(image)
    
    ort_inputs = {model.get_inputs()[0].name: input_image}
    ort_outs = model.run(None, ort_inputs)
    
    output = ort_outs[0][0]
    person_class_mask = (output.argmax(axis=0) == 15).astype('uint8')

    mask1 = person_class_mask * 255
    mask2 = (255-mask1)
    return mask1,mask2

# Use mask-rcnn model to find person segmentation
def findmask_mask_rcnn(image, model, threshold=0.5):
    image = image.transpose(-1, 0, 1)
    image = image/255
    input_image = np.array([image]).astype('float32')
    
    ort_inputs = {model.get_inputs()[0].name: input_image}
    ort_outs = model.run(None, ort_inputs)
    
    masks = (ort_outs[3]>0.5).squeeze()
    scores = ort_outs[2]
    
    valid_num = sum(map(lambda x : x > threshold, scores))
    if valid_num == 1 and len(scores) == 1:
        person_mask = masks.astype(np.uint8)
    elif valid_num > 0:
        person_mask = sum(masks[i] for i in range(len(masks)) if scores[i] > threshold).astype(np.uint8)
    else:
        person_mask = np.zeros(image.shape[:2], dtype='uint8')
    
    ret, mask1 = cv2.threshold(person_mask, 0, 255, cv2.THRESH_BINARY)
    mask2 = (255-mask1)
    return mask1,mask2