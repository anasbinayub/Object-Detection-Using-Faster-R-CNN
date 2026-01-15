# Object Detection using Faster R-CNN

## 📌 Overview

This project demonstrates **object detection using a pre-trained Faster R-CNN model** with a **ResNet-50 + Feature Pyramid Network (FPN)** backbone, implemented using **PyTorch and Torchvision**.  
The model is trained on the **COCO dataset** and performs object detection on images by drawing bounding boxes and class labels.

The implementation is designed to run in a **Google Colab environment** and performs **inference on a single image** stored in Google Drive.

---

## 🚀 Key Features

- Uses **Faster R-CNN (ResNet50-FPN)** pre-trained on COCO
- Performs **object detection inference**
- Applies **confidence threshold filtering**
- Visualizes **bounding boxes and class labels**
- Simple and easy-to-understand workflow

---

## 🧠 Model Information

- **Architecture:** Faster R-CNN  
- **Backbone:** ResNet-50 with FPN  
- **Dataset:** COCO (Common Objects in Context)  
- **Framework:** PyTorch + Torchvision  

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- PIL (Pillow)
- NumPy
- Google Colab

---

## 📂 Project Workflow

1. Import required libraries  
2. Load pre-trained Faster R-CNN model  
3. Set model to evaluation mode  
4. Mount Google Drive  
5. Load and preprocess input image  
6. Perform object detection inference  
7. Filter predictions using confidence threshold  
8. Draw bounding boxes and labels  
9. Display output image  

---

## 📸 Output Description

The model detects objects such as:

- Person
- Car
- Bicycle
- Boat
- Umbrella
- And many other COCO classes

Each detected object is displayed with:
- **Green bounding box**
- **Blue class label**

---

## 📁 COCO Classes

The project includes a predefined list of **COCO class names**, mapping class IDs to human-readable labels such as `person`, `car`, `dog`, `boat`, etc.

---

## ⚙️ How to Run the Project

### Option 1: Google Colab (Recommended)

1. Upload the notebook to Google Colab
2. Mount Google Drive
3. Place the input image inside Google Drive
4. Update the image path in the notebook
5. Run all cells

---

### Option 2: Local Machine

> ⚠️ This project is optimized for Google Colab. Minor changes may be required for local execution.

Install required libraries:
```bash
pip install torch torchvision opencv-python pillow
```

## 🧪 Confidence Threshold
```bash
scores > 0.2
```

Only predictions with confidence scores greater than **0.2** are displayed.
This value can be adjusted for stricter detection.

---

## ⚠️ Limitations

- No model training (inference only)
- Single image detection
- Uses pre-trained COCO weights
- Designed mainly for Google Colab

---

## 🔮 Future Enhancements

- Video object detection
- Batch image processing
- Custom dataset training
- Model fine-tuning
- Improved visualization
  
---

## 👤 Author

**Anas Bin Ayub**
  
---

## 📜 License

This project is intended for **educational and research purposes only.**
Pre-trained model weights belong to their respective owners.
  
---
