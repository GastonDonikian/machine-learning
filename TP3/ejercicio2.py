from sklearn import svm
from PIL import Image
import numpy as np
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Svm as s


def svm_classifier(x, y,test_size,output):
    # Create an SVM classifier
    clf = svm.SVC()
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    # Train the classifier
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    y_pred = clf.predict(output)
    return y_pred

def get_pixels(imagepath):
    img = Image.open(imagepath)
    im_matrix = np.array(img)
    return im_matrix.reshape(-1, 3)

def create_image_SVM(labels,width, height):
    # Example NumPy array with class labels
    # Define color map for each class
    color_map = {
    0: (0, 0, 255),   # Green
    1: (255, 0, 0),   # Red
    2: (0, 255, 0)  # Blue
    }
    
    # Create color image from class labels
    colors = np.array([color_map[label] for label in labels])
    colors = colors.reshape((height, width, 3))

    image = Image.fromarray(colors.astype('uint8'), 'RGB')

    # Save the image
    image.save("output.png")

if __name__ == '__main__':
    sky_matrix = get_pixels('./resources/cielo.jpg')
    cow_matrix = get_pixels('./resources/vaca.jpg')
    grass_matrix = get_pixels('./resources/pasto.jpg')
    cow = get_pixels('./resources/cow.jpg')
    image = Image.open('./resources/cow.jpg')
    width, height = image.size
    images = [sky_matrix, cow_matrix, grass_matrix]
    target_labels = [0, 1, 2]
    y = np.concatenate([np.full(p.shape[0], label) for p, label in zip(images, target_labels)])
    output = svm_classifier(np.concatenate(images), y,0.2,cow)   
    create_image_SVM(output,width,height)
    # dataset = []
    # dataset += [[x, 0] for x in sky_matrix]
    # dataset += [[x, 1] for x in cow_matrix]
    # dataset += [[x, 2] for x in grass_matrix]
    # svm_2 = s.SVM(max_epochs=100)
    # print(svm_2.svg_one_sample(dataset,3))
    
    