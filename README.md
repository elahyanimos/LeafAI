#  Plant Disease Classifier

A web application that uses deep learning to detect plant diseases from leaf images. Built with Streamlit and TensorFlow.

##  Features

- Upload or take photos of plant leaves
- Real-time disease classification
- Confidence score for predictions
- User feedback collection system
- Mobile-friendly interface


##  Requirements

- Python 3.10.9
- TensorFlow
- Streamlit
- PIL (Python Imaging Library)
- NumPy

##  Installation

1. Clone the repository:
```bash
git clone https://github.com/elahyanimos/LeafAI.git
cd LeafAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model file (`plant_disease_classifier.h5`) and place it in the `model/` directory.

##  Running the App

```bash
streamlit run plant_disease_app.py
```

The app will be available at `http://localhost:8501`

##  Docker Support

Build and run with Docker:

```bash
docker build -t plant-disease-classifier .
docker run -p 80:80 plant-disease-classifier
```

##  Usage

1. Upload an image using the sidebar
2. Wait for the analysis
3. View the predicted disease and confidence score
4. Provide feedback to help improve the model



