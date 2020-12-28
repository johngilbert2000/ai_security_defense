# Project Description

In this project, a deep learning model was trained to be robust
to adversarial examples (with epsilon <= 8 on a 0-255 scale)

To get predictions from the model, see the instructions below.

To see how the model was prepared, see the src/preparations/ folder.

More information can be found in report.pdf.


Update: This defense is vulnerable if the attacker knows the type of defense being used, and the weights stored in the repository could use more adversarial training targeting the updated model.

This was part of an assignment for an [AI security](https://www.csie.ntu.edu.tw/~stchen/teaching/spml20fall/index.html) course at NTU.



# Environment Setup

To setup an environment with TF 2.1, use:
```
pip install -r src/requirements.txt
```

Note that "h5py<3.0.0" may prevent a potential error with loading tf2cv models.

See this link for more details:
https://github.com/tensorflow/tensorflow/issues/44467


Alternatively, to use TF 2.3.1, use:
```
pip install -r src/requirements_other.txt
```

This project should work with both TF 2.1 and TF 2.3.1.


# Run Instructions

Put some 32x32 color .png images in 'example_folder/' 
or in another folder of your choosing.

(To generate some non-adversarial sample images, see the src/misc/ folder).
(To generate adversarial images, see the src/preparations/ folder).

To get model predictions, use the following: 

(If using Python >= 3.6)
```
python hw2.py example_folder/
```

(If using Python < 3.6)
```
python src/hw2_oldpython.py example_folder/
```

Output predictions will be saved to "predict.txt"

To check the accuracy of model predictions, see the src/misc/ folder.
