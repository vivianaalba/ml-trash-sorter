# ML Classification - Sorting Trash Images
This project builds an AI-powered image classifier to categorize waste into nine classes: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, and Vegetation. The goal is to develop a robust, generalizable model that performs well on unseen images while handling the natural variability in waste types.

## Summary
This project demonstrates how careful model selection, hyperparameter tuning, and data preparation can significantly improve image classification performance in real-world scenarios. I experimented with multiple deep learning architectures, including ResNet50, ResNet101, VGG16, and EfficientNetB0, applying techniques such as:
- Class balancing and label smoothing
- Fine-tuning hyperparameters like the number of ReLU layers and regularization strength
- Data augmentation (rotation, zoom, flips, translation, contrast adjustments)
- Different classification head architectures and dropout strategies

## Results
After evaluating the models on metrics including accuracy, F1 score, and AUC, EfficientNetB0 emerged as the best-performing model. It not only achieved the highest test accuracy and F1 score, but also showed strong generalization on unseen data. ResNet101 performed closely behind, followed by VGG16, and then ResNet50. <br>

| Model          | Accuracy (Train) | Precision (Train) | Recall (Train) | F1 (Train) | AUC (Train) | Accuracy (Val) | Precision (Val) | Recall (Val) | F1 (Val) | AUC (Val) | Accuracy (Test) | Precision (Test) | Recall (Test) | F1 (Test) | AUC (Test) |
|----------------|------------------|-------------------|----------------|------------|-------------|----------------|-----------------|--------------|----------|-----------|-----------------|------------------|---------------|-----------|------------|
| ResNet50       | 0.671269         | 0.666595          | 0.700302       | 0.673932   | 0.943781    | 0.552632       | 0.541804         | 0.545184     | 0.536800 | 0.892764  | 0.580000        | 0.558108         | 0.563008      | 0.556939  | 0.895267   |
| ResNet101      | **0.950362**     | **0.950173**      | **0.959713**   | **0.954564**| **0.998606**| 0.753947       | **0.762996**     | 0.739856     | 0.740293 | 0.965155  | 0.687368        | **0.699234**     | 0.669716      | 0.664182  | 0.948954   |
| EfficientNetB0 | 0.943458         | 0.942806          | 0.953403       | 0.947591   | 0.998269    | **0.814474**   | 0.810428         | **0.819432** | **0.807108**| **0.979132**| **0.751579**   | 0.741238         | **0.756339**  | **0.736158**| **0.963788**|
| VGG16          | 0.831032         | 0.829621          | 0.852226       | 0.838486   | 0.984378    | 0.718421       | 0.705191         | 0.717948     | 0.704300 | 0.948927  | 0.663158        | 0.652578         | 0.665190      | 0.652872  | 0.933046   |

## Technologies / Skills Used
- **Languages:** Python
- **Data Manipulation & Analysis:** pandas, numpy
- **Image Processing:** Pillow (PIL), TensorFlow image preprocessing
- **Deep Learning & Model Development:** TensorFlow/Keras (EfficientNetB0, ResNet101, VGG16, ResNet50), transfer learning, fine-tuning, dropout regularization, label smoothing, class balancing
- **Data Augmentation:** rotation, zoom, translation, flips, contrast adjustment, saturation adjustment
- **Model Evaluation & Metrics:** scikit-learn (accuracy, F1 score, AUC), train/test splitting, confusion matrices
- **Data Visualization:** matplotlib, seaborn
- **Other Skills:** reproducible ML pipelines, model performance comparison, hyperparameter tuning, generalization testing on unseen data

## Project Structure / How to Run
- `data/RealWaste` : contains folders that contain images of trash, used as training, test, and validation data
- `notebooks/` : contains Jupyter notebook ran by Google Colab
- `requirements.txt` : contains Python dependencies

## Requirements
numpy <br>
pandas <br>
matplotlib <br>
seaborn <br>
Pillow <br>
tensorflow <br>
scikit-learn <br>
