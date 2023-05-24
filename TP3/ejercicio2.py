from matplotlib import pyplot as plt
from sklearn import svm
from PIL import Image
import numpy as np
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Svm as s
from metrics import confusion_matrix, plot_confusion_matrix

kernels = ['poly','linear','sigmoid', 'rbf']

def svm_classifier(x, y, test_size, output, kernel):
    # Create an SVM classifier
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    original_x_train = x_train
    original_y_train = y_train
    accuracy_array = []
    linspace = np.linspace(0.001, 1, num=100)

    for i in linspace:
        clf = svm.SVC(kernel=kernel, max_iter=100, random_state=np.random.RandomState(42), C=i)
        # Split the data into training and testing sets
        x_train = original_x_train
        y_train = original_y_train
        # Train the classifier
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # Calculate the accuracy of the classifier
        accuracy = accuracy_score(y_test, y_pred)
        # plot_confusion_matrix(title=clf.kernel, conf_matrix=confusion_matrix([0, 1, 2], y_test, y_pred))
        print("Accuracy:", accuracy)
        accuracy_array.append(accuracy)
        # y_pred = clf.predict(output)

    return y_pred, accuracy_array, linspace


def get_pixels(imagepath, resize=1):
    img = Image.open(imagepath)
    size = img.size
    img = img.resize(np.array(size) // resize)
    im_matrix = np.array(img)

    return im_matrix.reshape(-1, 3)


def create_image_SVM(labels, width, height):
    # Example NumPy array with class labels
    # Define color map for each class
    color_map = {
        0: (0, 0, 255),  # Green
        1: (255, 0, 0),  # Red
        2: (0, 255, 0)  # Blue
    }

    # Create color image from class labels
    colors = np.array([color_map[label] for label in labels])
    colors = colors.reshape((height, width, 3))

    image = Image.fromarray(colors.astype('uint8'), 'RGB')

    # Save the image
    image.save("output.png")


if __name__ == '__main__':
    sky_matrix = get_pixels('./resources/cielo.jpg',resize=10)
    cow_matrix = get_pixels('./resources/vaca.jpg',resize=10)
    grass_matrix = get_pixels('./resources/pasto.jpg',resize=10)
    cow = get_pixels('./resources/cow.jpg',resize=10)
    image = Image.open('./resources/cow.jpg')
    width, height = image.size
    images = [sky_matrix, cow_matrix, grass_matrix]
    target_labels = [0, 1, 2]
    y = np.concatenate([np.full(p.shape[0], label) for p, label in zip(images, target_labels)])
    for i in kernels:
        output, accuracy_array, linspace = svm_classifier(np.concatenate(images), y, 0.2, cow, i)
        print(i)
        plt.plot(linspace, accuracy_array, '-', label=i)
    plt.xlabel("C value")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs C value")
    plt.legend()
    plt.legend()
    plt.grid()
    plt.savefig('./images/pasto_vaca_cielo_acc_vs_c_0_100.png', bbox_inches='tight')
    plt.show()
    # create_image_SVM(output,width,height)
    # dataset = []
    # dataset += [[x, 0] for x in sky_matrix]
    # dataset += [[x, 1] for x in cow_matrix]
    # dataset += [[x, 2] for x in grass_matrix]
    # svm_2 = s.SVM(max_epochs=100)
    # print(svm_2.svg_one_sample(dataset,3))
