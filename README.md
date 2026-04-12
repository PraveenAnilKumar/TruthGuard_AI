# 🛡️ TruthGuard AI
### Advanced Media Authenticity & Analysis Platform

TruthGuard AI is a state-of-the-art forensic tool designed to combat disinformation and verify media integrity. Built for the modern information landscape, it integrates deep learning models to detect deepfakes, verify news claims against live sources, and analyze the psychological impact of communication through sentiment and toxicity analysis.

---

## 🚀 Key Features

### 1. Deepfake Detection
*   **Image & Video Analysis**: Uses advanced CNN and Transformer architectures (MobileNet, ELA, FFT) to identify digital manipulations.
*   **Explainable AI**: Generates Error Level Analysis (ELA) heatmaps and Fast Fourier Transform (FFT) visualizations to show *where* a face was manipulated.

### 2. Fake News Detection
*   **Neural Verification**: Analyzes text patterns to determine the likelihood of misinformation.
*   **Live News Comparison**: Cross-references claims against a real-time index of mainstream news sources to find supporting or contradicting evidence.

### 3. Communication Analysis (Unified Scan)
*   **Sentiment Intelligence**: Detects emotional tone (Positive, Negative, Neutral) with high confidence.
*   **Toxicity Guard**: Identifies harmful content, hate speech, and harassment using an ensemble of safety models.
*   **Aspect Insights**: Breaks down sentiment by specific topics or entities mentioned in the text.

### 4. Interactive Visualizations
*   **Dynamic Gauges**: Real-time confidence scores for every detection.
*   **Emotional Histograms**: Visual breakdown of toxicity categories and sentiment trends.

---

## 🛠️ Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/) (High-performance web interface)
*   **Machine Learning**: TensorFlow, Keras, PyTorch, Scikit-Learn
*   **NLP**: Hugging Face Transformers, NLTK, TextBlob
*   **Computer Vision**: OpenCV, PIL, Facenet-PyTorch
*   **Data Visualization**: Plotly, Seaborn, Matplotlib

---

## 📦 Installation & Setup

### Prerequisites
*   Python 3.8+
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/PraveenAnilKumar/TruthGuard_AI.git
cd TruthGuard_AI
```

### 2. Set Up Environment
It is highly recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
ADMIN_PASSWORD=your_secure_password
ADMIN_REGISTRATION_KEY=your_key
# Optional: Add API keys for advanced news search if required
```

### 5. Download Models
Ensure the `models/` directory contains the required `.h5` and `.pth` files.

---

## 🚦 How to Run

Launch the application using Streamlit:
```bash
streamlit run app.py
```

---

## 🛡️ Responsible AI & Privacy
TruthGuard AI is designed for forensic and educational purposes. 
*   **Local Processing**: Models are designed to run locally to ensure data privacy.
*   **No Data Retention**: By default, uploaded media is processed in memory or stored in temporary directories that are cleared periodically.

---

## 👨‍💻 Contributor
**Praveen Anil Kumar**
*   GitHub: [@PraveenAnilKumar](https://github.com/PraveenAnilKumar)

---

## 📜 License
This project is for educational and research purposes. Please check the individual module headers for specific licensing information regarding the models used.
