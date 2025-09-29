# 📘 LungVision – Lung Cancer Detection System

## 📖 Overview
LungVision is a machine learning and deep learning project for multi-class classification of lung conditions (**Normal, Benign, Malignant**) using the **IQ-OTHNCCD lung cancer dataset**.  
The project compares three approaches: **Random Forest (baseline), Custom CNN (from scratch), and ResNet18 (transfer learning)**.  

---

## 📂 Dataset
- **Dataset**: [IQ-OTHNCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset)  
- **Preprocessing**:
  - Converted CT scan images to grayscale  
  - Resized images to fixed size (64×64 for RF, 128×128 for CNN/ResNet)  
  - Normalized pixel values to range [0, 1]  

---

## 🧠 Models Implemented
1. **Random Forest (Scikit-learn)**  
   - Baseline on flattened pixel features  
   - Fast training but limited accuracy  

2. **Custom CNN (PyTorch)**  
   - Built from scratch with convolutional & pooling layers  
   - Learns spatial features directly from images  

3. **ResNet18 (PyTorch, Transfer Learning)**  
   - Modified for grayscale input and 3-class output  
   - Achieved the best performance  

---

## 📊 Results
- **ResNet18** achieved the **highest accuracy and generalization**  
- Evaluated using:
  - Training & validation accuracy/loss curves  
  - Confusion matrix & classification report  
  - Grad-CAM heatmaps for interpretability  

---

## 📊 Model Comparison

| Model        | Accuracy | Training Time | Pros                                | Cons                                   |
|--------------|----------|---------------|-------------------------------------|----------------------------------------|
| Random Forest | ~70%     | Fast (CPU)    | Simple, no GPU required             | Limited image feature learning          |
| Custom CNN    | ~85%     | Moderate (GPU)| Learns spatial features automatically | Risk of overfitting on small datasets   |
| ResNet18      | ~95%+    | Slower (GPU)  | Transfer learning, best accuracy    | Larger model, requires more compute     |

---

## ⚙️ How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/LungVision.git
   cd LungVision
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run preprocessing & training:
   ```bash
   jupyter notebook LungVision.ipynb
   ```

---

## 🛠️ Technologies
- **PyTorch, Torchvision** (Deep Learning)  
- **Scikit-learn** (Random Forest)  
- **OpenCV, NumPy, Pandas** (Preprocessing)  
- **Matplotlib, Seaborn** (Visualization)  

---

## 📌 Future Work
- Experiment with ResNet50, DenseNet, or EfficientNet  
- Hyperparameter tuning for CNN  
- Deploy model with Flask/Streamlit for live demo  
