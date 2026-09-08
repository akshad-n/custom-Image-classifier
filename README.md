# 📷 Custom Photo Classifier

A deep learning image classification web application built with **TensorFlow / Keras** and **Streamlit**. It uses transfer learning with **MobileNetV2** to accurately classify technology and hardware objects.

---

## 🚀 Features

- **Deep Learning Model**: Transfer learning with MobileNetV2 (fine-tuned on 10 object classes).
- **Interactive Web Interface**: Streamlit UI with responsive dual-column layout.
- **Real-Time Predictions**: Instant classification with confidence metrics and Top-5 class probability breakdown.
- **Model Caching**: Cached model loading with `@st.cache_resource` for fast performance.

---

## 📋 Supported Classes

1. `USB stick`
2. `computer mouse`
3. `keyboard I`
4. `keys objects`
5. `laptop`
6. `magnifying glass`
7. `phone`
8. `router`
9. `satellite dish device`
10. `server rack`

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras**
- **Streamlit**
- **Pillow (PIL)**
- **NumPy**

---

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akshad-n/custom-Image-classifier.git
   cd custom-Image-classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   Or:
   ```bash
   python -m streamlit run app.py
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`.
