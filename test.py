from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def LSTM_model():
    max_len = 5
    model = Sequential()
    model.add(Embedding(input_dim=365, output_dim=8, input_length=max_len))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(365, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model(train_df, test_data):
    # Prepare training data
    X_train = train_df.iloc[:,:-1].values
    y_train = train_df.iloc[:,-1].values

    max_len = len(X_train)
    X_train_padded = pad_sequences(X_train, maxlen=max_len, padding='pre', dtype='float32')
    y_train_cat = to_categorical(y_train, num_classes=365)

    # Create and train the model
    model = LSTM_model()
    model.fit(X_train_padded, y_train_cat, epochs=5, batch_size=1, validation_split=0.2)

    # Prepare test data
    X_test = np.array(test_data)
    X_test_padded = pad_sequences(X_test, maxlen=max_len, padding='pre', dtype='float32')

    # Make predictions
    predictions = model.predict(X_test_padded)
    return predictions

# Example usage:
# Assuming you have a DataFrame 'df' with training data
# train_df = df
# test_data = [[-0.2, -0.1, 0.0]]
# predictions = model(train_df, test_data)
# print("Predictions:", predictions)

df = pd.read_csv('./Stepping_Right_DataFrame_with_Y_Column.csv')
train, test = train_test_split(df, train_size=0.7)
print(model(train, test))