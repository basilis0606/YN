import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.regularizers import l2, l1

# Data Preprocessing
# Read dataset
dataset = pd.read_csv('dataset-HAR-PUC-Rio.csv', delimiter=';', skiprows=(1))

# Encode categorical values as numeric
for column in dataset.columns:
    if dataset[column].dtype == type(object):
        dataset[column] = LabelEncoder().fit_transform(dataset[column])

# Features normalization
norm_dataset = StandardScaler().fit_transform(X=dataset)

# Split into input and output
X = norm_dataset[:, :-1]
Y = to_categorical(norm_dataset[:, -1])  # One hot encoding

learning_rates = [0.001, 0.001, 0.05, 0.1]
momentum_values = [0.2, 0.6, 0.6, 0.6]
r_values = [0.1, 0.5, 0.9]

hidden_layers_configs = [
    (10,),
    (10, 10),
    (10, 20),
    (20, 10),
    (10, 10, 10),
    (20, 10, 5),
    (5, 10, 20),
]

best_lr, best_momentum, best_r = None, None, None
best_hidden_layers_config = None
best_accuracy = 0

# Split the data using StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True)

for lr in learning_rates:
    for momentum in momentum_values:
        for r in r_values:
            for hidden_layers_config in hidden_layers_configs:
                mseList, ceList, accList, train_losses, val_losses = [], [], [], [], []
                # Early Stop
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                for train, test in skf.split(X, Y):
                    # Create model
                    model = Sequential()
                    model.add(Dense(hidden_layers_config[0], activation='relu', input_dim=17, kernel_regularizer=l2(r)))
                    
                    for nodes in hidden_layers_config[1:]:
                        model.add(Dense(nodes, activation='relu', kernel_regularizer=l2(r)))
                    
                    model.add(Dense(5, activation='softmax', kernel_regularizer=l2(r)))

                    # Compile model
                    optimizer = Adam(lr=lr, beta_1=momentum)
                    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', 'categorical_crossentropy', 'mean_squared_error'])

                    # Train the model
                    history = model.fit(X[train], Y[train], epochs=500, batch_size=500, verbose=0, validation_split=0.2, callbacks=[early_stopping])
                    train_losses.append(history.history['loss'])
                    val_losses.append(history.history['val_loss'])

                    # Evaluate model
                    scores = model.evaluate(X[test], Y[test], verbose=0)
                    loss, accuracy, categorical_crossentropy, mean_squared_error = scores
                    ceList.append(categorical_crossentropy)
                    mseList.append(mean_squared_error)
                    accList.append(accuracy)

                # Check if the current combination of learning rate, momentum, and hidden layers configuration results in a better accuracy
                current_accuracy = np.mean(accList)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_lr = lr
                    best_momentum = momentum
                    best_r = r
                    best_hidden_layers_config = hidden_layers_config

print(f"Average Cross-Entropy: {np.mean(ceList)}")
print(f"Average Mean Squared Error: {np.mean(mseList)}")
print(f"Average Accuracy: {np.mean(accList)}")

print(f"Best Learning Rate: {best_lr}")
print(f"Best Momentum: {best_momentum}")
print(f"Best Regularization Parameter (r): {best_r}")
print(f"Best Hidden Layers Configuration: {best_hidden_layers_config}")
print(f"Best Average Accuracy: {best_accuracy}")

mean_train_losses = np.mean(train_losses, axis=0)
mean_val_losses = np.mean(val_losses, axis=0)

plt.plot(range(1, len(mean_train_losses) + 1), mean_train_losses, label='Training')
plt.plot(range(1, len(mean_val_losses) + 1), mean_val_losses, label='Validation')
plt.show()
