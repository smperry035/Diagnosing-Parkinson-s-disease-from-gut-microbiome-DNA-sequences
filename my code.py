#importing 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

files = os.listdir()
dataset = pd.DataFrame()
for f in files:
    if f[0] == 'p':
        file = open(f)
        content = file.readlines()
        for i in range(1, len(content), 4):
            if content[i] != '':    
                p_patient = pd.Series([content[i], 1])
                p_row = pd.DataFrame([p_patient])
                dataset = pd.concat([dataset, p_row], ignore_index=True)
    if f[0] == 'c':
        file = open(f)
        content = file.readlines()
        for i in range(1, len(content), 4):
            if content[i] != '':
                c_patient = pd.Series([content[i], 0])
                c_row = pd.DataFrame([c_patient])
                dataset = pd.concat([dataset, c_row], ignore_index=True)
dataset.columns = ['Sequence', 'Status']

#creation of k-mers
def getKmers(sequence, size=4):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
dataset['Words'] = dataset.apply(lambda x: getKmers(x['Sequence']), axis=1)
pd.DataFrame(dataset).to_csv('X.csv')
print(dataset.iloc[110, 0])

#Bag-of Words model
S = 'actg'
bag = []
for a in S:
   for b in S:
      for c in S:
          for d in S:
              bag.append(a+b+c+d)

sparse = pd.DataFrame(np.zeros((len(dataset.index), 256)))
for index in dataset.index:
    kmer_list = list(dataset.loc[index, 'Words'])
    for kmer in kmer_list:
        try:
            location = bag.index(kmer)
            sparse.loc[index, location] = 1
        except ValueError:
            pass

#ANN
X = np.array(sparse)
y = np.array(dataset.iloc[:, 1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 16)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = ann.fit(X_train, y_train, batch_size = 32, epochs = 75)

#results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

plt.plot(history.history['acc'])
plt.title('Accuracy of Model vs. Epoch')
plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
plt.show()

#plot
import seaborn as sn

array = [[18, 2],
         [16, 5]]

df_cm = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
plt.title('Confusion Matrix', fontsize = 20)
plt.xlabel('Predicted', fontsize = 15) 
plt.ylabel('Actual', fontsize = 15) 

plt.show()