import os
import onnxruntime as rt
import cv2
import inference
from tqdm import tqdm

# Input the original image's path and use models to convert the image into a picture with cartoonize person
def get_output(img_path, deeplab, cartoon):      
    img = cv2.imread(img_path)
    img = inference.resize_crop(img)
    
    mask1,mask2 = inference.findmask_deeplabv3(img, deeplab)

    cartoon_img = inference.cartoonize(img, cartoon)
    
    person = cv2.bitwise_and(cartoon_img, cartoon_img, mask=mask1)
    background = cv2.bitwise_and(img, img, mask=mask2)
    output = person + background
    
    return output

if __name__=='__main__':
    # Load img's file names
    imgs = list(os.listdir(os.path.join('input_image')))
    
    # Load deeplab and cartoonize GAN model
    deeplab = rt.InferenceSession('deeplabv3.onnx',providers=['CPUExecutionProvider'])
    cartoon = rt.InferenceSession('cartoonize.onnx',providers=['CPUExecutionProvider'])
    
    path = "output_image"
    # Check whether the specified path exists or not
    if not os.path.exists(path):
       # Create a new directory because it does not exist
       os.makedirs(path)
       print("ouput_image directory is created!")
    
    # Process the input images and save them to the output folder
    for img in tqdm(imgs):
        output = get_output('input_image/'+img, deeplab, cartoon)
        cv2.imwrite('output_image/'+img.split('.')[0]+'.jpg', output)

    print('completed!')