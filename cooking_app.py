import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import Flask, request, redirect, render_template, url_for, flash



# Initialize Flask app
app = Flask(__name__, static_folder=r'D:\Paung Paung Project\CRAS\static',template_folder=r'D:\Paung Paung Project\CRAS\templates')
app.secret_key = 'your_secret_key'  # Replace with your secret key

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(app.root_path, r'D:\Paung Paung Project\CRAS\static\uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load the trained model
model_path = r'D:\Paung Paung Project\CRAS\cras_dataset_ep5.h5'
model = load_model(model_path)

# Directories for your training data (if needed for class names)
food_image_dir = r'D:\Paung Paung Project\CRAS\train_test_val_dataset\train_test_val_dataset\test'
base_dir = r'D:\Paung Paung Project\CRAS\train_test_val_dataset\train_test_val_dataset\test'
train_dir = r'D:\Paung Paung Project\CRAS\train_test_val_dataset\train_test_val_dataset\train'
val_dir = r'D:\Paung Paung Project\CRAS\train_test_val_dataset\train_test_val_dataset\val'
# Alternatively, if you have the classes in your train generator, adjust accordingly.
# Here we use the sorted list of folder names under the images directory.
class_indices = sorted(os.listdir(food_image_dir))

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Preprocess images for the chosen model
    rotation_range=30,                        # Increased rotation range for diverse orientations
    width_shift_range=0.2,                    # Simulate lateral movements of food
    height_shift_range=0.2,                   # Simulate vertical movements of food
    shear_range=0.2,                          # Simulate different angles
    zoom_range=0.3,                           # Increased zoom range for better scale variation
    horizontal_flip=True,                     # Flip images horizontally for better robustness
    fill_mode='nearest',                      # Fill empty spaces after transformations
    brightness_range=[0.8, 1.2],              # Simulate different lighting conditions
    channel_shift_range=20                    # Randomly change image colors
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input  # Only preprocessing for validation
)

# Creating the training and validation generators with the updated augmentations
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Ensure the images are resized to the input size of the model
    batch_size=16,           # Batch size of 16
    class_mode='categorical' # Multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),  # Resize images to model input size
    batch_size=16,           # Batch size of 16
    class_mode='categorical' # Multi-class classification
)
# Preprocess function for image
def preprocess(img):
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50
    return img_array


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash("No file part in the request")
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash("No file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Secure the filename and save it to the upload folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Load and preprocess the image using Keras
            try:
                img = image.load_img(file_path, target_size=(224, 224))
            except Exception as e:
                flash(f"Error loading image: {e}")
                return redirect(request.url)

            img_array = preprocess(img)

            # Predict the class of the image
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)

            # Get the predicted class from your class list
            try:
                predicted_class = list(train_generator.class_indices.keys())[np.argmax(predictions)]
            except IndexError:
                predicted_class = "Unknown"

            # Load the dataset for recipes
            csv_path = r'D:\Paung Paung Project\CRAS\myanmar food inclusive_Finalll.csv'
            df = pd.read_csv(csv_path, encoding='ISO-8859-1')

            # Filter the DataFrame by the predicted class (or title)
            result = df[df['Recipe Name'].str.contains(predicted_class, case=False, na=False)]

            # For demonstration, we print ingredients and instructions in the console.
            # You can pass these to your template as needed.
            recipe_info = {}
            recipe_info['predicted_class'] = predicted_class

            if not result.empty:
                ingredients = result['Ingredients'].values[0].split(',')
                recipe_info['ingredients'] = [ing.strip() for ing in ingredients]
                recipe_info['instructions'] = result['Instructions'].values[0]
            else:
                recipe_info['error'] = "Recipe not found."

            # Optionally, display the image prediction result using matplotlib

            # Save the figure if needed:
            # plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f'pred_{filename}'))
            plt.close()  # Close the plot to free memory

            # Render a template to display results
            return render_template('result.html', recipe=recipe_info,
                                   image_url=url_for('static', filename='uploads/' + filename))
        else:
            flash("Allowed file types are png, jpg, jpeg, gif")
            return redirect(request.url)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
