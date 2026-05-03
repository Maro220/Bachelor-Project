import cv2
from ultralytics import YOLO

def run_yolo_detection(image_path):
    print(f"Loading YOLOv8 model and analyzing {image_path}...")
    
    # 1. Load the pre-trained YOLOv8 'Nano' model (it's fast and lightweight)
    # Note: The first time you run this, it will take a few seconds to download the 'yolov8n.pt' weight file.
    model = YOLO('yolov8n.pt') 
    
    # 2. Run the image through the model
    # conf=0.25 means it will only show boxes it is at least 25% confident about
    results = model(image_path, conf=0.25)
    
    # 3. Ultralytics has a built-in '.plot()' function that magically draws 
    # all the bounding boxes, labels, and confidence scores onto your image!
    annotated_frame = results[0].plot()
    
    # 4. Show the image with bounding boxes
    print("👉 PRESS ANY KEY to close the image 👈")
    window_name = "YOLO Bounding Boxes"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    cv2.imshow(window_name, annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 5. Save the result
    output_name = "assets/yolo_output.jpg"
    cv2.imwrite(output_name, annotated_frame)
    print(f" Saved bounded image to {output_name}")

if __name__ == "__main__":
    # Point this to your webp image!
    my_image = "assets/14-street-lane-6799-1.webp" 
    
    run_yolo_detection(my_image)