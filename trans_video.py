import os
import onnxruntime as rt
import cv2
import inference
from tqdm import tqdm
from moviepy.editor import VideoFileClip

# Input the original image and use models to convert the image into a picture with cartoonize person
def get_output(img, deeplab, cartoon):      
    img = inference.resize_crop(img)
    
    mask1,mask2 = inference.findmask_deeplabv3(img, deeplab)

    cartoon_img = inference.cartoonize(img, cartoon)
    
    person = cv2.bitwise_and(cartoon_img, cartoon_img, mask=mask1)
    background = cv2.bitwise_and(img, img, mask=mask2)
    output = person + background
    
    return output

# Resize frame height and width to fit model output
def resize(h,w):
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    h, w = int((h//8)*8), int((w//8)*8)
    return (h, w)

if __name__=='__main__':
    # Load video's file names
    videos = list(os.listdir(os.path.join('input_video')))
    
    # Load deeplab and cartoonize GAN model
    deeplab = rt.InferenceSession('deeplabv3.onnx',providers=['CUDAExecutionProvider'])
    cartoon = rt.InferenceSession('cartoonize.onnx',providers=['CUDAExecutionProvider'])
    
    # Print ExecutionProvider
    device = {'CUDAExecutionProvider': 'GPU', 'CPUExecutionProvider': 'CPU'}
    print(f'using {device[cartoon.get_providers()[0]]} now')
    
    path = "output_video"
    
    # Check whether the specified path exists or not
    if not os.path.exists(path):
       # Create a new directory because it does not exist
       os.makedirs(path)
       print("ouput_video directory is created!")
    
    for video in videos:
        # Load the video to be processed and get the video parameters
        cap = cv2.VideoCapture('input_video/'+video)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        name = video.split('.')[0]
        
        # Set the output video parameters
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outfile = cv2.VideoWriter('output_video/temp_'+name+'.mp4',fourcc, fps, resize(width,height))
        
        print(f'video : {name} is transferring')
        
        # Transfer the video
        for _ in tqdm(range(int(frames))):
            ret, frame = cap.read()
            output = get_output(frame, deeplab, cartoon)        
            outfile.write(output)
        cap.release()
        outfile.release()
        
        # Get the sound of the original video
        out_video = VideoFileClip('output_video/temp_'+name+'.mp4')
        out_audio = VideoFileClip('input_video/'+video).audio
        
        # Synchronize sound and video and save the result
        output = out_video.set_audio(out_audio)
        output.write_videofile('output_video/'+name+'.mp4')
        os.remove('output_video/temp_'+name+'.mp4')

    print('completed!')
