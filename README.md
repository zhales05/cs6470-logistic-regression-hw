# CS 6470 - Logistic Regression Implementation

## Quick Start

### Prerequisites

- Python 3.7+
- Virtual environment (recommended)

### Installation & Setup

1. **Set up virtual environment**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Implementation

**Execute the main script:**

```bash
python main.py
```

## Metric comparision (also outputted after execution)

| Metric    | Custom Implementation | Scikit-learn | Difference |
| --------- | --------------------- | ------------ | ---------- |
| Accuracy  | 96.15%                | 100.00%      | 3.85%      |
| Precision | 100.00%               | 100.00%      | 0.00%      |
| Recall    | 92.31%                | 100.00%      | 7.69%      |
| ROC AUC   | 99.41%                | 100.00%      | 0.59%      |

**Analysis:** The custom implementation performs very close to sklearn's optimized version, with only 1 false negative out of 26 test samples.
