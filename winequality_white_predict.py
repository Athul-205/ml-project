import joblib as jb
import numpy as np
model = jb.load('whitewine_model.keras')
result = model.predict(np.array([[8.1,0.28,0.4,6.9,0.05,30,97,0.9951,3.26,0.44,10.1]]))
print("Predicted Quality=", result)