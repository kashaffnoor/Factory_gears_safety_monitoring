"""
Training script for PPE Detection using YOLOv8
"""

from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import TRAINING_CONFIG, DATASET_CONFIG, MODEL_CONFIG

def train_model():
    """Train YOLOv8 model on PPE dataset"""
    
    print("Starting PPE Detection Model Training...")
    print(f"Model: {MODEL_CONFIG['model_type']}")
    print(f"Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Image size: {TRAINING_CONFIG['image_size']}")
    
    # Check if dataset exists
    data_yaml = DATASET_CONFIG['data_yaml']
    if not os.path.exists(data_yaml):
        print(f"\n✗ Dataset configuration not found: {data_yaml}")
        print("\nPlease follow these steps:")
        print("1. Download a PPE dataset (see QUICK_START.md for options)")
        print("2. Place it in the data/ directory")
        print("3. Ensure data/data.yaml exists with correct paths")
        print("\nAlternatively, use the pre-trained model by running:")
        print("  streamlit run src/app.py")
        return None
    
    # Check if images exist
    train_images = Path(DATASET_CONFIG['train_path']) / 'images'
    if not train_images.exists() or len(list(train_images.glob('*.jpg'))) == 0:
        print(f"\n✗ No training images found in: {train_images}")
        print("\nPlease download and extract your dataset first.")
        print("See QUICK_START.md for dataset options.")
        return None
    
    # Initialize model with pre-trained weights
    # Using yolov8n.pt (nano version) for faster training and inference
    print(f"\nLoading {MODEL_CONFIG['model_type']}.pt...")
    model = YOLO(f"{MODEL_CONFIG['model_type']}.pt")
    
    # Train the model
    print("\nStarting training...")
    results = model.train(
        data=data_yaml,
        epochs=TRAINING_CONFIG['epochs'],
        batch=TRAINING_CONFIG['batch_size'],
        imgsz=TRAINING_CONFIG['image_size'],
        lr0=TRAINING_CONFIG['learning_rate'],
        patience=TRAINING_CONFIG['patience'],
        device=TRAINING_CONFIG['device'],
        project='runs/train',
        name='ppe_detection',
        exist_ok=True,
        pretrained=True,
        verbose=True
    )
    
    print("\nTraining completed!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    
    # Print training metrics
    print("\nTraining Results:")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    
    return results

if __name__ == "__main__":
    try:
        results = train_model()
        if results:
            print("\n✓ Training successful!")
        else:
            print("\n✗ Training skipped - no dataset found")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
