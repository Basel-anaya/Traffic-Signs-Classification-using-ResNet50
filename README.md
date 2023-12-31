# Traffic Signs Classification using ResNet50

![image](https://github.com/Basel-anaya/Traffic-Signs-Classification-using-ResNet50/assets/81964452/038928e3-aa9b-4990-949a-b83cb5c7c553)

This project aims to classify traffic signs using the ResNet50 deep learning model. It involves training a convolutional neural network (CNN) on a dataset of traffic sign images to learn to recognize and classify different types of traffic signs.

## Dataset

The dataset used for this project is the Traffic Signs Classification dataset, which consists of thousands of labeled images of traffic signs.

The dataset can be downloaded from the following source: [Traffic Sign Classification](https://www.kaggle.com/datasets/flo2607/traffic-signs-classification)

## Model Architecture

The ResNet50 model architecture is used for traffic sign classification. ResNet50 is a deep convolutional neural network architecture that has shown excellent performance on a wide range of computer vision tasks. It consists of 50 layers, including residual blocks, which help in training deeper networks and mitigating the vanishing gradient problem.

## Dependencies

The following dependencies are required to run the project:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install the dependencies using pip:

```bash
pip install tensorflow keras numpy matplotlib
```

## Usage 

1. Clone the repository:

```bash
git clone https://github.com/Basel-anaya/Traffic-Signs-Classification-using-ResNet50.git
```

2. Download the dataset and extract it into a directory named `dataset`.

3. Open the `traffic-signs-classifications-using-resnet50.ipynb` file and set the desired hyperparameters, such as `batch size`, `number of epochs`, and `learning rate`.

4. Run the code cells

5. After training, the model will be saved as `ResNet_model.h5`.

6. To evaluate the model on the `test set`, run this script:
```python
# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Plot the predictions
plt.figure(figsize=(10, 10))
for i in range(25):  # Adjust the number of samples to plot as needed
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.title(f"Predicted: {y_pred_classes[i]}, Actual: {np.argmax(y_test[i])}")

plt.tight_layout()
plt.show()
```

## Results

The results achieved by the trained `ResNet50 model` on the test set are as follows:

- Test accuracy: `99.2%`

### Some plots

#### Training plot
![line_Training_plot](https://github.com/Basel-anaya/Traffic-Signs-Classification-using-ResNet50/assets/81964452/242a13b4-c71f-45eb-a3b0-303b1cab46e9)

#### Accuracy Plot
![line_Accuracy_plot](https://github.com/Basel-anaya/Traffic-Signs-Classification-using-ResNet50/assets/81964452/9ee9a229-bd12-4050-83d7-490a682540b4)

#### Loss Plot
![line_Loss_plot](https://github.com/Basel-anaya/Traffic-Signs-Classification-using-ResNet50/assets/81964452/9b26fec2-79a1-4b76-a588-c8dc908a9ce6)

#### Confusion Matrix
![cm_plot](https://github.com/Basel-anaya/Traffic-Signs-Classification-using-ResNet50/assets/81964452/de85ba19-4d0c-4cd2-82c5-5720d7a51a24)


## License

This project is licensed under the [Apache 2.0 License](LICENSE).
