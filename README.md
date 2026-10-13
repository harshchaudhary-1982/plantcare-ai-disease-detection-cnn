# Results folder

After training and evaluation, this folder contains actual training curves, a CSV history, test metrics, a classification report, and a confusion-matrix plot. Generated outputs are ignored by Git to avoid publishing unverified or run-specific results. Add only results you actually produced and explain the dataset/split used.

 
 project details:

Item	Details
Project name	PlantCare AI
Developer	Harsh Chaudhary
Programming language	Python
Deep learning framework	TensorFlow/Keras
Model	Convolutional Neural Network (CNN)
Web application	Streamlit
Image processing	Pillow and NumPy
Evaluation	scikit-learn and Matplotlib
GitHub repository	plant-disease-detection-cnn

Current status: The project files have been created and basic code checks have been performed. The dataset has not been downloaded, and the CNN has not yet been trained. Consequently, no actual disease-classification accuracy is available yet.
The project has the following structure:

PlantCare-AI/

│
├── app.py

├── train.py

├── evaluate.py

├── predict.py

├── config.py

├── requirements.txt

├── README.md

├── LICENSE


├── .gitignore
│

├── data/

│   └── README.md
│
├── models

│   └── README.md

│
├── results/

│   └── README.md

│
├── src/

│   ├── __init__.py

│   ├── data_loader.py

│   ├── model.py

│   ├── preprocessing.py

│   └── utils.py
│
├── tests/

│   ├── test_preprocessing.py

│   └── test_prediction.py
│
└── docs/

    ├── project_overview.md
    
    └── methodology.md
    File	Purpose
app.py	:Creates the Streamlit web application.

train.py:	Splits the dataset and trains the CNN.

evaluate.py	:Calculates test accuracy, precision, recall, and F1-score.

predict.py :Predicts the class of an uploaded image from the command line.

config.py:	Stores common paths and training settings.

src/model.py	:Defines the CNN architecture.

src/data_loader.py	:Discovers image classes and creates dataset splits.

src/preprocessing.py	:Resizes and converts images to RGB.

requirements.txt	:Lists the Python dependencies.

README.md:	Explains setup, training, evaluation, and GitHub publishing.

The code also includes training and validation graphs, a confusion matrix, saved class names, and automated tests.


