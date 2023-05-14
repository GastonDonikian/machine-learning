from sklearn import svm
from PIL import Image
import numpy as np
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def svm_classifier(x, y):
    # Create an SVM classifier
    clf = svm.SVC()
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Train the classifier
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def get_pixels(imagepath):
    img = Image.open(imagepath)
    im_matrix = np.array(img)
    return im_matrix.reshape(-1, 3)

if __name__ == '__main__':
    sky_matrix = get_pixels('./resources/cielo.jpg')
    cow_matrix = get_pixels('./resources/vaca.jpg')
    grass_matrix = get_pixels('./resources/pasto.jpg')

    images = [sky_matrix, cow_matrix, grass_matrix]
    target_labels = [0, 1, 2]
    y = np.concatenate([np.full(p.shape[0], label) for p, label in zip(images, target_labels)])
    svm_classifier(np.concatenate(images), y)

    #pixel_list = list(map(lambda x: (x,0),sky_matrix))
    #pixel_list.append(list(map(lambda x: (x,1),cow_matrix)))
    #pixel_list.append(list(map(lambda x: (x,2),grass_matrix)))
    #print(pixel_list)
    #svm_clasifier(pixel_list)

    
    