# Medibuddy: Smart Disease Predictor

## Modular PyTorch Training & File Structure

### Project Structure

- `config/train_config.yaml`: Central config for all models and training parameters
- `src/training/`: Modular training scripts for each model
  - `train_malaria.py`, `train_pneumonia.py`, `train_tabular.py`, `train_all.py`
- `src/utils/preprocess.py`: Centralized preprocessing and model architecture logic
- `data/`: Automated storage for CSV datasets
- `models/`: Saved models (.pth for PyTorch, .pkl for scikit-learn)
- `requirements.txt`: Python dependencies (PyTorch, CUDA, scikit-learn, etc.)

### Possible Commands

#### 1. Setup & Data
- **Install Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
- **Download Tabular Datasets**:
  ```bash
  python data/download_datasets.py
  ```

#### 2. Training Models
- **Train Everything**:
  ```bash
  python src/training/train_all.py
  ```
- **Train Tabular Models Only**:
  ```bash
  python src/training/train_all.py --tabular
  ```
- **Train Image Models Only (PyTorch + CUDA)**:
  ```bash
  python src/training/train_all.py --image
  ```
- **Train Individual Model (e.g., Diabetes)**:
  ```bash
  python src/training/train_diabetes.py
  ```

#### 3. Get Training Results (Terminal + PNG)
- During training, terminal output includes:
  - Test Accuracy
  - Classification Report
  - Confusion Matrix values
- A confusion matrix PNG is also saved in `models/` for each trained model.

- **Example: run all models and get all result artifacts**
  ```bash
  python src/training/train_all.py
  ```

- **List saved confusion matrix PNG files**
  ```bash
  ls models/*confusion_matrix.png
  ```

- **Expected confusion matrix files**
  - `models/diabetes_confusion_matrix.png`
  - `models/heart_confusion_matrix.png`
  - `models/breast_cancer_confusion_matrix.png`
  - `models/kidney_confusion_matrix.png`
  - `models/liver_confusion_matrix.png`
  - `models/malaria_confusion_matrix.png`
  - `models/pneumonia_confusion_matrix.png`

#### 4. Run Web Application
- **Start Flask Server**:
  ```bash
  python app.py
  ```

### Logging
- Training scripts use `tqdm` for live progress bars in the terminal.
- Major logs are also saved to the `models/` directory for debugging.

### Web Application
- Flask app for disease prediction (tabular and image models)
- HTML templates in `templates/`
- Static assets in `static/`

### Sample images of the web application

#### Home Page
<img src="images/Sample_Web_App_Images/sample1.png" alt="Home"/>
<br>

#### Diabetes Predictor
<img src="images/Sample_Web_App_Images/sample2.png" alt="Diabetes"/>
<br>

#### Breast Cancer Predictor
<img src="images/Sample_Web_App_Images/sample3.png" alt="Breast Cancer"/>
<br>

#### Malaria Predictor
<img src="images/Sample_Web_App_Images/sample4.png" alt="Malaria"/>
<br>

#### Negative Result Page
<img src="images/Sample_Web_App_Images/sample5.png" alt="Negative Result"/>
<br>

#### Positive Result Page
<img src="images/Sample_Web_App_Images/sample6.png" alt="Positive Result"/>
<br>
