Video Link: https://youtu.be/eBrGyuA2MIg


# 🌟 **What is Normalization in Machine Learning?**

**Normalization** is a feature scaling technique that transforms numerical values into a **common scale**, typically **0 to 1**.

It keeps the **shape of the data distribution**, but compresses it so no feature dominates others.

Normalization is also called:

*   **Min‑Max Scaling**
*   **Rescaling**
*   **0–1 Scaling**

***

# 🔢 **Normalization (Min–Max Scaling) Formula**

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

Where:

*   $$x$$ = original value
*   $$x_{min}$$ = minimum value of the feature
*   $$x_{max}$$ = maximum value of the feature
*   $$x'$$ ∈ $$[0,1]$$

***

# 🎯 Why Do We Use Normalization?

Normalization is used when:

*   Features have **different scales**
*   Using distance‑based models:
    ✔ K‑Means  
    ✔ KNN  
    ✔ Neural networks  
    ✔ Logistic Regression

Example:
If **income** = 200000 and **age** = 30,
the model may consider *income 6000 times more important* unless scaled.

***

# 🧠 **Simple Example (Step-by-Step Calculation)**

Consider the feature:

    Salaries = [20,000, 50,000, 100,000]

### Step 1: Identify min & max

*   Min = 20,000
*   Max = 100,000

### Step 2: Apply Min-Max formula

For value = 50,000:

$$
x' = \frac{50000 - 20000}{100000 - 20000}
$$

$$
x' = \frac{30000}{80000} = 0.375
$$

### 📌 Final Normalized Values

| Original | Normalized |
| -------- | ---------- |
| 20,000   | 0.0        |
| 50,000   | 0.375      |
| 100,000  | 1.0        |

Now the entire feature lies in **\[0,1]**.

***

# 🖼 Visual Intuition (Conceptual)

Before normalization:

    20,000 ————————————— 100,000

After normalization:

    0.0 —————— 0.3 —————— 1.0

Same order, smaller scale.

***

# 🧪 **Python Example: Min‑Max Normalization**

```
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame({
    'salary': [20000, 50000, 80000, 100000]
})

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['salary']])

print(df_scaled)
```

### Output:

    [[0.   ]
     [0.375]
     [0.75 ]
     [1.   ]]

***

# 🧩 When Should You Use Normalization?

### Use normalization when:

✓ You use **distance-based** models  
✓ Data has **different units** (km, kg, $, years)  
✓ Neural networks — helps faster convergence  
✓ Using gradient descent-based models

### Do NOT normalize when:

✗ Using **tree-based** models  
(Random Forest, XGBoost, LightGBM)

Trees do not care about scale.


# 🌍 Real‑World Example: Normalizing Image Pixels

Images have pixel values:

    0–255

Deep learning models **always normalize** images:

$$
x' = \frac{x}{255}
$$

This speeds up training and stabilizes gradients.

Example:

```
image = image / 255.0
```

***

# 🏁 Final Summary

**Normalization** rescales features to the range **\[0,1]**, ensuring all features contribute equally.

### You should use it when:

*   Features have different units
*   Using KNN, K‑Means, Neural Networks, Logistic Regression
*   Preparing image data for CNNs

### Formula:

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

***


