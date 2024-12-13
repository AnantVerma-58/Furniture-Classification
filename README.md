# Furniture Image Classification

A web-based application that allows users to classify furniture images using multiple pre-trained models. The app offers the flexibility to upload your own image of any extension and size or choose from a set of sample images. It is powered by Streamlit and is designed to showcase the capabilities of deep learning in image classification.

## Features

1. **Multiple Model Options**:
   - Users can select from multiple trained models to classify furniture images.
   - Models include popular architectures trained on custom furniture datasets (Densenet201,VGG16, Resnet50).

2. **Image Upload**:
   - Upload your own furniture image for classification.
   - Supported file formats: `.jpg`, `.png`, `.jpeg`.

3. **Sample Images**:
   - Choose from a variety of pre-loaded sample images to test the models.

4. **Custom Dataset**:
   - Models were trained on a custom dataset of furniture images collected and annotated first-hand.
   - Data augmentation techniques were used to enhance the dataset for improved performance.

5. **Deployment**:
   - Deployed on [Streamlit Cloud](https://furniture-classification.streamlit.app/).

## How It Works

1. **Select a Model**:
   - Choose a model from the dropdown menu.

2. **Upload or Select an Image**:
   - Upload an image file or select a sample image provided in the app.

3. **View Prediction**:
   - Click the "Predict" button to view the classification result.
   - The predicted furniture class is displayed along with confidence scores.

## Models Used

The following models were trained and deployed:

- **DenseNet**
- **ResNet**
- **Custom CNN**

These models were trained on a custom dataset of furniture images and tested for accuracy and reliability.

## Dataset

- **Collection**:
  - Images were collected from various sources and annotated manually.
- **Augmentation**:
  - Data augmentation techniques such as rotation, flipping, and scaling were applied to increase dataset diversity.
- **Training and Testing**:
  - The models were trained on a split of the dataset with a rigorous testing phase to ensure high performance.

## Installation (For Local Deployment)

1. Clone the repository:
   ```bash
   git clone https://github.com/AnantVerma-58/Furniture-Classification.git
   cd Furniture-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deployment

The app is deployed on Streamlit Cloud and can be accessed [here](https://furniture-classification.streamlit.app/).

## File Structure

```
/Furniture-Classification
├── .gitattributes
├── app.py
├── augment.py
├── densenet.pt
├── evaluate.py
├── model.py
├── pred.py
├── prediction.py
├── requirements.txt
├── resnet.pt
├── run.py
├── split.py
├── train.py
├── transform.py
├── trials.ipynb
├── utils.py
├── vgg.pt
|-- README.md             # Project documentation
```

## Requirements

- Python 3.12
- Libraries: `torch`, `torchvision`, `opencv-python-haedless`, `streamlit`, `numpy`

## Future Improvements

- Add more pre-trained models for classification.
- Expand the dataset to include more furniture categories.
- Implementing real-time prediction using a webcam.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---
