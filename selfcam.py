import onnxruntime as rt
import cv2
import inference

def get_output(img, deeplab, cartoon):      
    img = inference.resize_crop(img)
    
    mask1,mask2 = inference.findmask_deeplabv3(img, deeplab)

    cartoon_img = inference.cartoonize(img, cartoon)
    
    person = cv2.bitwise_and(cartoon_img, cartoon_img, mask=mask1)
    background = cv2.bitwise_and(img, img, mask=mask2)
    output = person + background
    
    return output

if __name__=='__main__':
    deeplab = rt.InferenceSession('deeplabv3.onnx',providers=['CUDAExecutionProvider'])
    cartoon = rt.InferenceSession('cartoonize.onnx',providers=['CUDAExecutionProvider'])
    
    device = {'CUDAExecutionProvider': 'GPU', 'CPUExecutionProvider': 'CPU'}
    print(f'using {device[cartoon.get_providers()[0]]} now')
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("can't open camera")
        exit()
        
    while True :
        ret, frame = cap.read()
        
        if not ret:
            print("can't receive frame")
            break
            
        output = get_output(frame, deeplab, cartoon)
    
        cv2.imshow('cartoon selfcam', output)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()