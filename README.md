# Dimensionality Reduction and Classification on CIFAR-10 Dataset using PCA

## 🚀 Project Overview

This project applies **Principal Component Analysis (PCA)** for dimensionality reduction on the CIFAR-10 image dataset, followed by classification using several machine learning models:  
- Random Forest Classifier  
- Support Vector Classifier (SVC)  
- Decision Tree Classifier

The goal is to reduce the feature dimensions while preserving 95% of variance, then train classifiers for image classification.

---

## 📚 Dataset: CIFAR-10

- 60,000 color images of size 32x32 pixels in 10 classes.  
- 50,000 training images and 10,000 test images.  
- Classes include: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

---

## 📊 Data Visualization

```python
fig = plt.figure(figsize=(8,8))
for i in range(64):
    ax = fig.add_subplot(8,8,i+1)
    ax.imshow(images_train[i], cmap=plt.cm.bone)
plt.show()
```



Visualizing 64 sample images from the training set.

⚡ PCA for Dimensionality Reduction

Reshape images into 1D vectors: (50000, 3072) for training and (10000, 3072) for testing.

Fit PCA and select number of components that explain 95% of variance.
```Python
pca = PCA()
pca.fit(data_train)

# Find optimal k
k = 0
total_variance = sum(pca.explained_variance_)
current_sum = 0
while current_sum / total_variance < 0.95:
    current_sum += pca.explained_variance_[k]
    k += 1
```

# Apply PCA with optimal components
```Python
pca_t = PCA(n_components=k, whiten=True)
x_train_pca = pca_t.fit_transform(data_train)
x_test_pca = pca_t.transform(data_test)
```

🌟 Classification Models
✅ Random Forest Classifier (RF)
```Python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

clf_RF = RandomForestClassifier()
grid_RF = {
    'n_estimators': [5, 10, 15, 20],
    'max_depth': [5, 10, 20, 30]
}
grid_search_RF = GridSearchCV(clf_RF, grid_RF)
grid_search_RF.fit(x_train_pca, cls_train)
```

# Best Random Forest Model
```Python
grid_search_RF.best_estimator_
grid_search_RF.best_score_
```

✅ Support Vector Classifier (SVC)
```Python
from sklearn import svm
from sklearn.model_selection import KFold

clf_SVC = svm.SVC()
grid_SVC = {
    'C': [1e2, 1e3, 5e3, 1e4],
    'gamma': [1e-3, 5e-4, 1e-4]
}
grid_search_SVC = GridSearchCV(
    clf_SVC, grid_SVC, 
    cv=KFold(n_splits=10, shuffle=True, random_state=1)
)
grid_search_SVC.fit(x_train_pca, cls_train)

grid_search_SVC.best_score_
```

✅ Decision Tree Classifier (DT)
```Python
from sklearn.tree import DecisionTreeClassifier

clf_DT = DecisionTreeClassifier(criterion="entropy", splitter="best")
grid_DT = {
    'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
    'max_depth': [10, 15, 20, 25, 30, 35],
    'min_impurity_decrease': [1e-2, 1e-3, 1e-4]
}

grid_search_DT = GridSearchCV(
    clf_DT, grid_DT,
    cv=KFold(n_splits=10, shuffle=True, random_state=1)
)
grid_search_DT.fit(x_train_pca, cls_train)

grid_search_DT.best_score_
```

✅ Conclusion

PCA significantly reduces dimensionality while preserving most of the dataset's variance.

Grid Search improves model performance by tuning hyperparameters.

Random Forest, SVC, and Decision Tree are suitable classifiers for reduced-dimension image data.
