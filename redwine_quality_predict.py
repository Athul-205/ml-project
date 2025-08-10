#RED WINE PREDICTION

import joblib as jb
import numpy as np
model = jb.load('redwine_model.keras')
result = model.predict(np.array([[11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8]]))
print("Predicted Quality=", result)