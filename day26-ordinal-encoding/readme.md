Video Link: https://youtu.be/w2GglmYHfmM


# 🌟 **What is Encoding Categorical Data?**

Machine learning models work best when data is **numerical**.  
But many real‑world datasets contain **categorical** (text-based) data, such as:

*   Gender → Male, Female
*   City → Pune, Mumbai, Delhi
*   Size → Small, Medium, Large
*   Education → High School, Graduate, Post‑Graduate

**Encoding** converts these text categories into **numbers**, so ML algorithms can understand them.

***

# 🎯 Why Do We Encode Categorical Data?

Because ML algorithms **cannot process text** directly.

Encoding helps:

*   Convert text to numbers
*   Preserve relationships (if any)
*   Improve model accuracy
*   Maintain consistent input format

***

# 🧩 Types of Categorical Encoding

Here are the most common forms:

1.  **Label Encoding**
2.  **Ordinal Encoding**
3.  One‑Hot Encoding
4.  Target Encoding
5.  Binary Encoding
6.  Hash Encoding

In this explanation, we focus on **Label Encoding** and **Ordinal Encoding**.

***

# 1️⃣ **Label Encoding**

Label Encoding converts **each unique category** into a **unique numeric value**.

👉 Use label encoding when:

*   Categories **do NOT have any order**
*   Example: color, city, gender, country

### ✔ Example

    Color:
    Red, Blue, Green

Label Encoding assigns:

| Color | Encoded |
| ----- | ------- |
| Red   | 0       |
| Blue  | 1       |
| Green | 2       |

⚠ But note:  
Models may *think* that **Green (2) > Blue (1)** → **order bias**  
This is why label encoding is NOT good for tree algorithms?  
Actually it is fine for trees, but NOT for **linear models, SVM, KNN**.

***

### ✔ Python Example (LabelEncoder)

```
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Red']
})

le = LabelEncoder()
df['Color_Encoded'] = le.fit_transform(df['Color'])

print(df)
```

Output:

       Color  Color_Encoded
    0    Red              2
    1   Blue              0
    2  Green              1
    3    Red              2

***

# 2️⃣ **Ordinal Encoding**

Ordinal Encoding is used when categories have a **natural order**.

👉 Use ordinal encoding when:

*   Categories are **ranked**
*   There is a **meaningful progression**

### ✔ Examples:

*   Size → Small < Medium < Large
*   Education → High‑School < Graduate < Postgraduate
*   Ratings → Poor < Average < Good < Excellent

Ordinal encoding respects this **order**.

***

### ✔ Example

    Size:
    Small, Medium, Large

Ordinal Encoding:

| Size   | Encoded |
| ------ | ------- |
| Small  | 0       |
| Medium | 1       |
| Large  | 2       |

🚀 This “order” helps models understand that **Large > Medium > Small**.

***

### ✔ Python Example (OrdinalEncoder)

```
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

df = pd.DataFrame({
    'Size': ['Small', 'Large', 'Medium', 'Small']
})

encoder = OrdinalEncoder(categories=[['Small','Medium','Large']])
df['Size_Encoded'] = encoder.fit_transform(df[['Size']])

print(df)
```

Output:

        Size  Size_Encoded
    0  Small           0.0
    1  Large           2.0
    2 Medium           1.0
    3  Small           0.0

✔ Order preserved  
✔ Model understands ranking

***

# 📌 Label Encoding vs Ordinal Encoding — Quick Comparison

| Feature        | Label Encoding         | Ordinal Encoding             |
| -------------- | ---------------------- | ---------------------------- |
| Used for       | Unordered categories   | Ordered categories           |
| Example        | Color                  | Size                         |
| Values Meaning | Arbitrary              | Meaningful                   |
| Model Impact   | May create false order | Correctly respects order     |
| Best For       | Tree models            | Any model needing order info |

***

# 🎬 Real‑World Example: Titanic Dataset

| Column              | Type          | Encoding Type                 |
| ------------------- | ------------- | ----------------------------- |
| Sex                 | Male/Female   | Label Encoding                |
| Embarked            | S, C, Q       | Label Encoding                |
| Pclass              | 1,2,3         | Already numeric but *ordinal* |
| Cabin (letter only) | A,B,C,D,E,F,G | Ordinal (if deck ranking)     |

***

# 🧠 Summary

### ✔ Encoding Categorical Data

Converting text data → numeric so ML models can process it.

### ✔ Label Encoding

*   Assigns numbers (0,1,2…)
*   Only for **unordered categories**

### ✔ Ordinal Encoding

*   Converts ordered categories
*   Preserves rank


