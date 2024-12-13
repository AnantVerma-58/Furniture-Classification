# Furniture Image Classification

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/torch-2.5.1-orange)
![TorchVision](https://img.shields.io/badge/torchvision-0.20.1-green)
![NumPy](https://img.shields.io/badge/numpy-2.2.0-violet)
![Streamlit](https://img.shields.io/badge/streamlit-1.41.0-brightgreen)
![OpenCV](https://img.shields.io/badge/opencv--python--headless-4.10.0.84-purple)


![Docker Pulls](https://img.shields.io/docker/pulls/anant58/furniture-classification)
![Docker Image Size](https://img.shields.io/docker/image-size/anant58/furniture-classification/1.3)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-orange)](https://furniture-classification.streamlit.app/)

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
   - Docker Image Available on [Docker Hub](https://dockerhub.com)

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
- **VGG**

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

## Docker Image

This application is also available as a Docker image, making it easy to run locally without setting up the environment manually.

- **Docker Hub Repository**: [Furniture Image Classification](https://hub.docker.com/r/anant58/furniture-classification)
- **Pull the Image**:
  ```bash
  docker pull anant58/furnitureimage:1.3
  docker run -d -p 9999:9999 anant58/furniture-classification:1.3
- **Acess The app a t [http://localhost:9999](http://localhost:9999)**

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
