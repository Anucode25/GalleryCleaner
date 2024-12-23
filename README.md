# Automated Image Classification and Meme Cleanup

This project offers a streamlined Python-based solution to automate the classification of images into "Genuine" or "Meme" categories. It leverages deep learning with MobileNetV2 for efficient binary classification and includes functionality to move meme images into a designated "Trash" folder while preserving genuine images.

## Key Features
- **Deep Learning Integration**: Uses a pre-trained MobileNetV2 model fine-tuned for binary image classification.
- **Automation**: Automatically processes images stored in a directory and organizes them by classification.
- **User-Friendly Cleanup**: Meme images are moved to a separate trash folder, reducing manual effort.
- **Performance Insights**: Visualizes training and validation accuracy, and calculates the time taken for the entire process.
- **Robust Error Handling**: Ensures graceful handling of edge cases like empty directories or unreadable images.

## How It Works
1. **Data Preprocessing**: Images are resized to 224x224 pixels and normalized for model input.
2. **Training**: The MobileNetV2 model is trained on labeled meme and genuine images with an 80-20 train-validation split.
3. **Prediction**: New images are classified, with memes moved to a specified trash folder.
4. **Visualization**: Training progress is visualized with accuracy graphs.
5. **Summary**: Provides statistics on the number of images processed, memes moved, and total execution time.

## Requirements
- Python 3.8+
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Pre-processed images for training and validation

## Usage
1. Clone the repository and set up the required directories for meme and genuine images.
    ```bash
    git clone https://github.com/your-repository-name.git
    cd your-repository-name
    ```

2. Modify paths in the script to match your directory structure.
3. Run the script to train the model, classify images, and organize them:
    ```bash
    python image_classifier.py
    ```

## Output
- A trained model that classifies images as genuine or meme.
- Meme images moved to the specified "Trash" folder.
- A summary of the number of images classified, memes moved, and time taken.
- Graphs visualizing training and validation accuracy.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request for any improvements.


---

Happy coding! 🚀
