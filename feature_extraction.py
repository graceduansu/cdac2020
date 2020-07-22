import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deepscribe import DeepScribe


def extract_features(save_path):
    """
    symbol_dict = {}
    DeepScribe.load_images(symbol_dict)
    print("symbol_dict len (number of symbols):", len(symbol_dict))
    DeepScribe.transform_images(symbol_dict, gray=False, resize=100)
    img_data = []

    for symb_name in symbol_dict:
        for symb_img in symbol_dict[symb_name]:
            img_data.append(symb_img.img)

    img_data = np.array(img_data, dtype='float32')
    
    np.save("output/img_data_color.npy", img_data)
    """
    img_data = np.load("output/img_data_color1.npy")
    print(img_data.shape)

    # Forgot test-train split here?
    x = keras.applications.vgg16.preprocess_input(img_data)
    model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    print("begin prediction:")
    features = model.predict(x, verbose=1)
    print(features)
    
    np.save(save_path, features)


def neural_net():
    img_data = np.load("output/img_data_color1.npy")
    print(img_data.shape)
    X = keras.applications.vgg16.preprocess_input(img_data)
    y = np.load("output/label_data_gray.npy")
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    predictions = keras.layers.Dense(232, activation='softmax')(x)

    print("basemodel input:", base_model.input)
    model = tf.keras.models.Model(inputs=base_model.input, ouptuts=predictions)

    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy', "top_k_categorical_accuracy"])
    
    cp = tf.keras.callbacks.ModelCheckpoint("vgg16_nn_{epoch:02d}_{loss:.2f}.hdf5", verbose=1, monitor="loss", save_freq=5)
    history = model.fit(x=x_train,y=y_train, epochs=50, verbose=1, callbacks=[cp])
    print(history)


def random_forest(X, y):
    # Reshape X for sklearn
    # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
    
    X = np.reshape(X, [X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]])
    print("X shape:", X.shape)

    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    """
    rf = RandomForestClassifier(verbose=2)
    rf.fit(x_train, y_train)
    
    pickle.dump(rf, open(save_path, 'wb'))
    """
    save_path = "output/vgg16_random_forest.sav"
    rf = pickle.load(open(save_path, 'rb'))

    y_pred = rf.predict(x_test)

    # Get symbol names MUST RUN WITH -L MAX
    symbol_names = list(DeepScribe.count_symbols(sort_by_alpha=True))
    print(symbol_names)

    y_test_list = []
    # Transform y values to corresponding label names
    for i in range(len(y_test)):
        s_idx = int(y_test[i])
        y_test_list.append(symbol_names[s_idx])
    
    y_pred_list = []
    for i in range(len(y_pred)):
        s_idx = int(y_pred[i])
        y_pred_list.append(symbol_names[s_idx])

    print("y_test_list:", y_test_list)
    print()
    print("y_pred_list:", y_pred_list)

    print(classification_report(y_test_list, y_pred_list))

    """
    # Print top 1 accuracy
    print("Top 1 accuracy:")
    print(accuracy_score(y_test, y_pred))

    # Print top 5 accuracy
    # TODO: Why is top 5 lower than top 1?
    print("Top 5 accuracy:")
    n = 5
    probas = rf.predict_proba(x_test)
    print("probas shape:", probas.shape)

    successes = 0
    total = 0
    
    for i in range(len(y_test)):
        top_n_predictions = np.argpartition(probas[i], -n)[-n:]
        # print("top n:", top_n_predictions)
        # print("ytest[i]:", y_test[i])

        if y_test[i] in top_n_predictions:
            # print("success")
            successes += 1
        total +=1
    print(successes/total)
    """