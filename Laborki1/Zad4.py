import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


data = pd.read_csv("train.csv")

data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})
data["Embarked"] = data["Embarked"].fillna(0)


X = data.drop(columns=["Survived"])
y = data["Survived"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(epochs=10, batch_size=32, learning_rate=0.001):
    model = keras.Sequential([
        layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n✅ Dokładność modelu: {acc*100:.2f}%")


train_model(epochs=15, batch_size=32, learning_rate=0.005)
