# Compares accuracy of predictions in predict.txt 
# with true labels stored in another file (correct.txt)

import numpy as np

PREDICTIONS_FILE = "../../predict.txt"
TRUE_LABELS_FILE = "../../correct.txt"

predictions = open(PREDICTIONS_FILE, "r")
p = predictions.read()
predictions.close()

correct = open(TRUE_LABELS_FILE, "r")
c = correct.read()
correct.close()

p_arr = np.array(p.split("\n")[:-1])
c_arr = np.array(c.split("\n")[:-1])

length = len(p_arr) if (len(p_arr) < len(c_arr)) else len(c_arr)

accuracy = np.mean(p_arr[:length] == c_arr[:length])

print(f"{length} predictions checked")
print(f"Accuracy: {accuracy:.2f} %")
