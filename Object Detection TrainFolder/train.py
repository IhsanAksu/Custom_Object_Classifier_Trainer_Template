from ultralytics import YOLO

def main():
    # Load the model
    model = YOLO('yolo26n.pt') 

    # Train the model
    results = model.train(
        data=r'dataset\data.yaml', 
        epochs=50, 
        imgsz=640, 
        name='my_tool_model',
        mode='train' # Explicitly set the mode
    )

if __name__ == '__main__':
    main()