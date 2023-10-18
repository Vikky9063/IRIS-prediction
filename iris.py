#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load the Iris dataset
data = pd.read_csv('IRIS2.csv')


# Select the petal length (feature) and the target (species)
X = data[['sepal_length']]
y = data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression(solver='liblinear', multi_class='auto')

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# You can now use the trained model to predict the species of iris flowers based on petal length
# For example, to predict the species of a flower with a petal length of 4.5 cm:
new_data = np.array([[3.4]])
predicted_species = model.predict(new_data)
print("Predicted Species:", [predicted_species][0])


# In[2]:


print(data.head())


# In[3]:


print(data.tail())


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt


species_counts = data['species'].value_counts()
sns.set(style="whitegrid")
sns.barplot(x=species_counts.index, y=species_counts.values)
plt.xlabel('Species')
plt.ylabel('Count')
plt.title('Iris Species Distribution')
plt.show()


# In[7]:


import matplotlib.pyplot as plt

# Example data
categories = ['Category A', 'Category B', 'Category C']
values = [10, 15, 7]

plt.bar(categories, values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.show()


# In[8]:


import matplotlib.pyplot as plt

# Example data
categories = ['Category A', 'Category B', 'Category C']
values = [10, 15, 7]

plt.barh(categories, values)
plt.xlabel('Values')
plt.ylabel('Categories')
plt.title('Horizontal Bar Chart Example')
plt.show()


# In[9]:


import matplotlib.pyplot as plt

# Assuming you have a dataset you want to plot as a histogram
plt.hist(data['sepal_length'], bins=10, edgecolor='k')  # Adjust the number of bins as needed
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.title('Sepal Length Histogram')
plt.show()


# In[ ]:




