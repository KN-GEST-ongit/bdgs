# BDGS development setup

## Prerequisities

- Python 3.11
- PIP

## Installation

### BDGS library

1. Navigate to `./bdgs` directory
2. Install required packages with `pip install -r requirements.txt`

### BDGS test scripts app

1. Navigate to `./bdgs_test_scripts` directory
2. Install BDGS library with `pip install ../bdg`
3. Install the rest of required packages with `pip install -r requirements.txt`
4. Run selected script (`main.py`)

## Usage

To create a new algorithm, follow these steps:

1. Navigate to `./bdgs/bdgs/algorithms` directory
2. Create a new package named with your algorithm name
3. In newly created directory create `.py` file named with your algorithm name
4. The file should contain a class that inherits from `BaseAlgorithm` class and implements `classify` method, as follows:

   ```python
   from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
   from bdgs.gesture import GESTURE


   class Alg2(BaseAlgorithm):
       def classify(self, image) -> GESTURE:
           return GESTURE.OK
   ```

   The method should take `image` parameter - an instance of openCV `Mat` class and return `GESTURE` - value of an enum which contains all gestures recognizable by the BDGS library.

5. Add newly created method to enum and dict in `./bdgs/bdgs/classifier.py` as follows:

   ```python
   class ALGORITHM(StrEnum):
       ALG_1 = "ALG_1"
       ALG_2 = "ALG_2" #added


   ALGORITHM_FUNCTIONS = {
       ALGORITHM.ALG_1: Alg1(),
       ALGORITHM.ALG_2: Alg2(), #added
   }
   ```

To load the changes in BDGS library into test scripts app, navigate to `./bdgs_test_scripts` directory and run `pip install --ignore-installed ../bdgs`.
