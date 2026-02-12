***

# ⭐ **What Is Standardization in Machine Learning?**

**Standardization**, also called **Z‑Score Normalization**, is a feature scaling technique that transforms data so that:

*   **Mean = 0**
*   **Standard Deviation = 1**
*   Distribution becomes centered and scaled

This helps ML models learn better because all features have comparable scales.

***

# 🔢 **Standardization Formula**

$$
z = \frac{x - \mu}{\sigma}
$$

Where:

*   $$x$$ = original value
*   $$\mu$$ = mean of the feature
*   $$\sigma$$ = standard deviation

After standardization:

*   Values close to mean → become near **0**
*   Higher values → become **positive**
*   Lower values → become **negative**

***

# 📘 **Simple Example (Step-by-Step Calculation)**

Consider this feature:

    Age = [20, 22, 24, 26, 28]

### Step 1: Calculate Mean

$$
\mu = \frac{20 + 22 + 24 + 26 + 28}{5} = 24
$$

### Step 2: Calculate Standard Deviation

$$
\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{n}} = 2.828
$$

### Step 3: Apply Z‑score formula

| Value (x) | z = (x – 24)/2.828        |
| --------- | ------------------------- |
| 20        | (20–24)/2.828 = **−1.41** |
| 22        | −0.71                     |
| 24        | 0                         |
| 26        | +0.71                     |
| 28        | +1.41                     |

### 📌 Final Standardized Values

    [-1.41, -0.71, 0, 0.71, 1.41]

Now the data is centered around **0**.

***

# 🧠 **Why Standardization Works?**

Many ML models rely on:

*   **Distance** (KNN, K-means)
*   **Gradient descent optimization** (Logistic Regression, Neural Networks)
*   **Variance** (PCA)

If one feature (e.g., income = 500,000) has larger values than others (e.g., age = 25), it **dominates** the model.

Standardization ensures **equal influence** among features.

***

# 🖥️ **Python Example: Standardizing Two Features**

```
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    'age': [20, 22, 24, 26, 28],
    'salary': [20000, 25000, 40000, 60000, 80000]
})

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

print(scaled_df)
```

### Output (example)

    [[-1.41, -1.23],
     [-0.71, -0.91],
     [ 0.00, -0.10],
     [ 0.71,  0.97],
     [ 1.41,  1.27]]

Both features are now scaled similarly.

***

# 🎯 **When Should You Use Standardization?**

### Recommended For:

*   Logistic Regression
*   Linear Regression
*   SVM
*   KNN
*   K-Means
*   PCA
*   Neural Networks

### Not Needed For:

*   Decision Trees
*   Random Forest
*   XGBoost / LightGBM  
    (because they are scale‑invariant)

***

# 🎬 Real‑World Example: Standardizing Height & Weight

Raw Data:

    Height (cm): 150–190
    Weight (kg): 50–120

Weight has a smaller numeric range.

Without scaling:

*   Height dominates distance-based models (KNN)
*   Neural network gradients become unstable
*   SVM decision boundaries skew

After standardization:

*   Both features contribute equally
*   Model trains faster
*   Accuracy improves

***

# 🏁 **Summary**

**Standardization makes all features comparable** by shifting them to zero mean and unit variance.

✔ Prevents domination by large-scale features  
✔ Improves ML model performance  
✔ Essential for distance-based and gradient-based algorithms  
✔ Simple to compute and widely used

***

