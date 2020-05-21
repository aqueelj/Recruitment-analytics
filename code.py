

# libraries

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense


# load the dataset

# The dataset contains information about students in an MBA program and if they got recruited on campus and if they
#did , then with what salary

path = 'C:/Users/user/Desktop/Files 2/Code/College Placement/Placement_Data_Full_Class.csv'
df = pd.read_csv(path, header=None)

#Drop Salary column for classification problem

#df_cl= df.drop([0,14],axis=1)

df.columns = df.iloc[0]
df = df[1:]


pd.set_option('display.max_columns', None)
df.head()

X=df.iloc[:,1:12]
y=df.iloc[:,13]

X.head()

#separating numerical and categorical columns and encoding categorical columns. Both dfs are then concatenated.
 

X_num=df[['ssc_p','hsc_p','degree_p','etest_p','mba_p']]

X_cat=X[['gender','degree_t','workex','specialisation']]


X_cat=pd.get_dummies(X_cat)

X=pd.concat([X_cat, X_num], axis=1)


# split into input and output columns
X, y = X.values, y.values

# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)



# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features

n_features = X_train.shape[1]

# define model


x_in = Input(shape=(14,))
outputs_1 = Dense(10,activation='relu',kernel_initializer='he_normal')(x_in)
outputs_2 = Dense(5,activation='relu',kernel_initializer='he_normal')(outputs_1)
x_out = Dense(1,activation='sigmoid')(outputs_2)
# define the model
model = Model(inputs=x_in, outputs=x_out)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1)

model.save('C:/Users/user/Desktop/Files 2/Code/College Placement/')

# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)




# make a prediction
row = [1,0,0,0,1,1,0,1,0,90,85,90,92,90]
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)

