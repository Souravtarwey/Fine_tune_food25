**Food-101 Image Classification**

**Objective**
The objective of this project is to develop a robust image classification model using a subset of 25 random classes from the Food-101 dataset. Tasks include preprocessing the dataset, exploring data characteristics through EDA, cleaning erroneous labels, implementing a train/test split, utilizing various pretrained feature extraction models (e.g., InceptionV3, ResNet50), adding necessary classification layers, performing hyperparameter tuning, augmenting data for improved model generalization, and visualizing feature maps of the finalized classifier.

**Food classes used here are**
Selected classes: ['pulled_pork_sandwich', 'chicken_wings', 'ravioli', 'pizza', 'tuna_tartare', 'miso_soup', 'beignets', 'caprese_salad', 'dumplings', 'chocolate_cake', 'garlic_bread', 'foie_gras', 'onion_rings', 'scallops', 'hot_and_sour_soup', 'shrimp_and_grits', 'omelette', 'eggs_benedict', 'spaghetti_bolognese', 'club_sandwich', 'frozen_yogurt', 'french_toast', 'fried_calamari', 'fried_rice', 'gnocchi']


**Libraries Used**
TensorFlow: Deep learning framework for model development and training.
Keras: High-level API for building and training deep learning models.
NumPy: Fundamental package for numerical computing.
Pandas: Data manipulation and analysis library for structured data operations.
Matplotlib: Plotting library for data visualization.
PIL: Python Imaging Library for image manipulation.
Scikit-learn: Machine learning library for data preprocessing and utility functions.
Keras Tuner: Library for hyperparameter tuning of Keras models.

**Project Breakdown**
**1. Dataset Preparation and Exploration**
Subset Selection: Randomly chose 25 classes from Food-101 dataset for focused analysis.
Exploratory Data Analysis (EDA): Visualized class distributions, image sizes, and other statistics to understand data characteristics.

**2. Data Preprocessing**
Data Cleaning: Corrected mislabeled images to ensure data integrity.
Train/Test Split: Divided dataset into training and validation sets for model evaluation.
Data Augmentation: Applied augmentation techniques (rotation, shifting, flipping) to training data to enhance model robustness.

**3. Model Development**
Feature Extraction: Employed pretrained models (e.g., InceptionV3, ResNet50) for feature extraction.
Model Architecture: Added custom classification layers (Conv2D, GlobalAveragePooling2D, Dense, Dropout) on top of feature extractors.
Fine-Tuning: Unfroze last 10 layers of base model and fine-tuned on augmented data for enhanced performance.

**4. Hyperparameter Tuning**
Conducted extensive hyperparameter tuning experiments to optimize model performance:
Adjusted learning rates, batch sizes, optimizer configurations, dropout rates, and layer architectures.

**5. Visualization**
Feature Maps: Visualized activation maps to interpret model behavior and feature extraction capabilities.
Instructions


**Environment Setup**
Ensure Python environment with necessary libraries (requirements.txt).
Consider using Google Colab or a local setup with GPU support for faster computation.

Reproducing Results
Follow the provided code snippets for dataset preprocessing, model building, training, and evaluation.
Adjust parameters and experiment configurations as needed based on project goals and computational resources.

**Conclusion**

