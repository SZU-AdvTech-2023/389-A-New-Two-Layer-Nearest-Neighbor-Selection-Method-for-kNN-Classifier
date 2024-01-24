# kTLNN

## 1. Pre-preparation
- **Python Version:** 3.9
- **Required Environment:**
  - `tqdm`

## 2. Run the Code

### 2.1 `main.py`
To run this code with 120 (k, k_b) pairs, modify the data corresponding to the following variables in `main.py`:

```python
file_path = './dataset/glass.txt'   # the path of dataset
multirun = 10    # number of repeated runs
column = 10    # number of columns of labeled columns (starting from 1)
save_path = 'result_Ionosphere.txt'    # where to save your calculations
```
The results will be saved in a txt file, you can change the name of 'save_path' to achieve it.

### 2.2 `test.py`
You can run `test.py` directly to get the results for a specific dataset for a specific (k, k_b) pair by modifying the following

```python
file_path = './dataset/Dermatology.txt'    # the path of dataset
multirun = 5   # number of repeated runs
column = 35  # number of columns of labeled columns(starting from 1)
k = 1
rate = 1.0
kb = int(k * rate)   # # rounding operation
```
The result will be printed directly in the terminal.
