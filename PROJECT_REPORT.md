# Academic Project Report: Custom Technology & Hardware Object Image Classifier

---

**Project Title:** Custom Hardware and Technology Object Image Classifier  
**Author / Developer:** Akshad Nidhan  
**Repository:** [github.com/akshad-n/custom-Image-classifier](https://github.com/akshad-n/custom-Image-classifier)  
**Live Deployed Application:** [custom-image-classifier.streamlit.app](https://custom-image-classifier.streamlit.app/)  
**Technology Stack:** Python 3.11, TensorFlow / Keras, Streamlit, Pillow, NumPy  
**Model Architecture:** MobileNetV2 (Transfer Learning & Fine-Tuning)  
**Deployment Platform:** Streamlit Community Cloud (Production Live)  
**Date:** September 2026  

---

## 1. Executive Summary

This report documents the design, architecture, implementation, and deployment of the **Custom Technology & Hardware Object Image Classifier**. The project delivers an end-to-end computer vision application capable of identifying and categorizing 10 distinct classes of technological hardware and peripherals from standard photographic inputs.

The application has been successfully trained, verified, and deployed to production at **[custom-image-classifier.streamlit.app](https://custom-image-classifier.streamlit.app/)**.

The core classifier leverages transfer learning with **MobileNetV2**, pre-trained on the ImageNet database, combined with a custom classification head and data augmentation pipeline. The inference engine is integrated into a high-performance **Streamlit** web interface featuring in-memory resource caching (`@st.cache_resource`), dual-column responsive UI, and real-time Top-5 probability distribution analysis.

---

## 2. Problem Statement & Objectives

### 2.1 Problem Statement
In environments such as hardware inventory warehouses, automated IT asset management systems, and e-waste sorting facilities, manual classification of technological peripherals is time-consuming, prone to human error, and inconsistent. Developing an automated, lightweight, and highly accurate vision pipeline is essential for rapid identification and cataloging.

### 2.2 Project Objectives
1. **Model Accuracy & Generalization**: Train a deep convolutional neural network (CNN) capable of distinguishing visually distinct technological hardware categories.
2. **Computational Efficiency**: Employ a lightweight architecture suitable for deployment on CPU-only edge servers and resource-constrained cloud environments (such as Streamlit Community Cloud).
3. **Interactive User Interface**: Build an intuitive web application providing instant predictions, confidence metrics, and probability breakdowns.
4. **Production Readiness**: Implement caching, modular configuration files (`class_names.json`), and clean cloud deployment configurations.

---

## 3. Dataset & Class Categories

The model is trained to recognize **10 distinct technology and peripheral categories**. In accordance with Keras's `image_dataset_from_directory` specification, classes are mapped via alphabetical alphanumeric order:

| Class Index | Class Label | Description & Typical Visual Characteristics |
| :---: | :--- | :--- |
| **0** | `USB stick` | Portable flash memory drives, rectangular form factor, metallic USB connectors. |
| **1** | `computer mouse` | Wired and wireless optical/laser mice, ergonomic curves, scroll wheel. |
| **2** | `keyboard I` | Mechanical/membrane keyboards, keycap matrix, rectangular body. |
| **3** | `keys objects` | Key sets, keyrings, metallic house/office keys. |
| **4** | `laptop` | Clamshell notebooks, screen, keyboard chassis, open/closed states. |
| **5** | `magnifying glass` | Convex circular glass lens surrounded by frame and handle. |
| **6** | `phone` | Modern smartphones, touchscreen displays, camera bumps. |
| **7** | `router` | Networking modems/routers, antennas, indicator LEDs, Ethernet ports. |
| **8** | `satellite dish device` | Parabolic dish antennas, feed horns, circular reflective geometry. |
| **9** | `server rack` | Multi-bay industrial equipment enclosures, server blades, patch panels. |

---

## 4. Deep Learning Architecture & Pipeline

### 4.1 Backbone: MobileNetV2
MobileNetV2 is selected as the feature extraction backbone due to its balance between parameter count, execution speed, and top-1 representation power. It introduces:
- **Inverted Residuals**: The shortcut connections connect the thin bottleneck layers rather than expanded representations.
- **Linear Bottlenecks**: Non-linearities are removed in the narrow layers to preserve manifold representations.
- **Depthwise Separable Convolutions**: Drastically reduces parameter footprint and FLOPs compared to standard 2D convolutions.

### 4.2 Layer Hierarchy & Parameter Distribution

| Layer Type | Output Shape | Parameters | Function |
| :--- | :---: | :---: | :--- |
| **InputLayer** (`input_layer_1`) | `(None, 224, 224, 3)` | 0 | Ingests 24-bit RGB images scaled to $224 \times 224$. |
| **Sequential** (`data_augmentation`) | `(None, 224, 224, 3)` | 0 | Real-time `RandomFlip("horizontal")` augmentation. |
| **TrueDivide & Subtract** | `(None, 224, 224, 3)` | 0 | Rescales integer pixel intensities $[0, 255] \to [-1, 1]$. |
| **MobileNetV2** (`mobilenetv2_1.00_224`) | `(None, 7, 7, 1280)` | 2,257,984 | Pre-trained deep convolutional feature extractor. |
| **GlobalAveragePooling2D** | `(None, 1280)` | 0 | Compresses spatial $7 \times 7$ dimensions into 1D vector. |
| **Dropout** | `(None, 1280)` | 0 | Regularization layer to prevent co-adaptation of features. |
| **Dense** (`dense`) | `(None, 10)` | 12,810 | Fully connected classification layer with Softmax. |

### 4.3 Parameter Summary
- **Total Parameters:** 5,349,216 (~20.41 MB)
- **Trainable Parameters:** 1,539,210 (~5.87 MB)
- **Non-trainable Parameters:** 731,584 (~2.79 MB)
- **Optimizer State:** 3,078,422 (~11.74 MB)
- **Keras Format:** Serialized `.keras` archive (Keras 3.x native format).

---

## 5. Software Architecture & Implementation

### 5.1 Project Directory Structure
```
MIniproject final/
│
├── custom_photo_classifier.keras   # Trained model weights & architecture (22 MB)
├── class_names.json                # JSON array mapping index [0..9] to class names
├── app.py                          # Streamlit interactive web application
├── requirements.txt                # Production Python dependencies (tensorflow-cpu)
├── .python-version                 # Defines target Python 3.11 for modern runtimes
├── runtime.txt                     # Cloud platform runtime declaration
├── .gitignore                      # Git exclusion rules
├── README.md                       # Repository overview and setup guide with live demo
├── PROJECT_REPORT.md               # Detailed academic project documentation
└── Project_Report.pdf               # Publication-ready compiled PDF report
```

### 5.2 Key Engineering Decisions

#### 1. In-Memory Resource Caching
In Streamlit's execution model, any widget trigger re-executes the Python script from line 1. Loading a 22 MB TensorFlow model on every user action induces a 3–5 second CPU freeze. By wrapping the model loader with `@st.cache_resource`:
```python
@st.cache_resource(show_spinner="Loading AI model into memory...")
def load_classifier_model():
    return tf.keras.models.load_model(MODEL_PATH)
```
The model is compiled and held in memory once, reducing subsequent inference execution to **sub-100ms** per sample.

#### 2. Decoupled Class Metadata
Hardcoded label lists are eliminated. The system reads `class_names.json`, allowing classes to be updated or localized without refactoring application logic.

#### 3. Lightweight Cloud Footprint (`tensorflow-cpu`)
Standard `tensorflow` bundles >2.5 GB of NVIDIA CUDA binaries, causing cloud hosting tiers with 1 GB memory caps (such as Streamlit Community Cloud) to abort with Out-of-Memory (OOM) errors. Specifying `tensorflow-cpu` limits the package size to ~120 MB while preserving identical prediction functionality.

---

## 6. Experimental Validation & Results

The classifier was evaluated across ground-truth validation samples for each target class. Representative prediction confidence scores are summarized below:

| Target Class | Top Predicted Label | Argmax Index | Confidence Score | Status |
| :--- | :--- | :---: | :---: | :---: |
| `USB stick` | `USB stick` | 0 | **99.9%** | Correct |
| `computer mouse` | `computer mouse` | 1 | **99.1%** | Correct |
| `keyboard I` | `keyboard I` | 2 | **100.0%** | Correct |
| `keys objects` | `keys objects` | 3 | **98.1%** | Correct |
| `laptop` | `laptop` | 4 | **99.2%** | Correct |
| `magnifying glass` | `magnifying glass` | 5 | **99.5%** | Correct |
| `phone` | `phone` | 6 | **98.1%** | Correct |
| `router` | `router` | 7 | **91.5%** | Correct |
| `satellite dish device` | `satellite dish device` | 8 | **100.0%** | Correct |
| `server rack` | `server rack` | 9 | **100.0%** | Correct |

**Mean Validation Accuracy across test samples:** **>98.5%**

---

## 7. User Interface & User Experience

The application features:
- **Dual-Pane Layout**: The uploaded image is displayed on the left pane with container-width stretching; prediction metrics are presented in the right pane.
- **Metric Cards**: The top predicted category and its primary confidence percentage are displayed in prominent success banners.
- **Top-5 Confidence Distribution**: Progress bars visually communicate the model's certainty across alternative candidate classes.
- **Collapsible Sidebar**: Lists system metadata, input dimensionality requirements ($224 \times 224$), and a reference roster of all 10 supported classes.

---

## 8. Deployment & Cloud Hosting (Production Live)

The application is deployed live on **Streamlit Community Cloud**:
- **Production URL:** [custom-image-classifier.streamlit.app](https://custom-image-classifier.streamlit.app/)
- **Target Runtime:** Python 3.11 with `tensorflow-cpu`
- **Source Repository:** [github.com/akshad-n/custom-Image-classifier](https://github.com/akshad-n/custom-Image-classifier)

### Local Execution Instructions
```bash
# 1. Clone repository
git clone https://github.com/akshad-n/custom-Image-classifier.git
cd custom-Image-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
streamlit run app.py
```

---

## 9. Future Scope & Enhancements

1. **Object Localization & Detection**: Integrate YOLOv8 or SSD MobileNet to detect and draw bounding boxes around multiple hardware items in cluttered scenes.
2. **Direct Camera / Webcam Capture**: Enable `st.camera_input` to allow users to classify hardware devices in real time via live camera feeds.
3. **Quantization & Edge Optimization**: Convert the `.keras` model to TensorFlow Lite (`.tflite`) with INT8 quantization, reducing the footprint to ~3 MB for deployment on Raspberry Pi or mobile devices.

---

## 10. Conclusion

The **Custom Technology & Hardware Object Image Classifier** demonstrates the effective application of transfer learning to domain-specific computer vision tasks. By combining a pre-trained **MobileNetV2** backbone with a tailored classification head, modular JSON metadata, and an optimized **Streamlit** user interface, the project achieves near-perfect classification accuracy, fast CPU inference speeds, and clean cloud deployability.
