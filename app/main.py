from fastapi import FastAPI
import numpy as np
from svm_lib import SVM
import time
import json

app = FastAPI()

@app.post("/svm")
async def vv(elementos_por_clase: int):
   
    start = time.time()

    # Generate some linearly separable data
    np.random.seed(0)

    # Features for class -1
    X_neg = np.random.randn(elementos_por_clase, 2) - 2 * np.ones((elementos_por_clase, 2))

    # Features for class 1
    X_pos = np.random.randn(elementos_por_clase, 2) + 2 * np.ones((elementos_por_clase, 2))

    # Combine features and create labels
    X = np.vstack((X_neg, X_pos))
    y = np.hstack((-np.ones(elementos_por_clase), np.ones(elementos_por_clase)))

    # Instantiate SVM model
    model = SVM(0.01, 0.01, 1000)

    # Train the model
    model.fit(X, y)

    # Predict on the training set
    predictions = model.predict(X)

    # Calculate accuracy
    accuracy = np.mean(predictions == y)
    str0 = f"{accuracy * 100:.2f}%"

    end = time.time()

    var1 = 'Time taken in seconds: '
    var2 = end - start

    str1 = f'{var1}{var2}'.format(var1=var1, var2=var2)
    
    data = {
        "Accuracy": str0,
        "Time taken": str1
    }
    jj = json.dumps(data)
    
    return jj