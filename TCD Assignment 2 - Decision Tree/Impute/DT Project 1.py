#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from sklearn import tree
from IPython.display import Image 
from sklearn.externals.six import StringIO 
from sklearn.tree import export_graphviz
#from pydot import graph_from_dot_data
import pydotplus

# Function to create Decision Tree
def create_DT(X,Y,name):
    print ("\n\n============================ "+name+" ==============================\n")
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 70)

    dtc = DecisionTreeClassifier(max_depth=5)
    model = dtc.fit(X_train, y_train)



    y_pred = dtc.predict(X_test)
    #print (y_pred)
    print ("* Accuracy : ", accuracy_score(y_test,y_pred)*100, "%")
    print ("\n* Confusion Matrix")
    print ("-------------------------")
    print (confusion_matrix(y_test,y_pred))
    print ("\n* Classification Report")
    print ("-------------------------")
    print (classification_report(y_test,y_pred))

    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, filled=True, rounded=True,class_names=['0','1'], impurity=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("E:/Trinity/Data Analytics/Assignment 2 - Decision Tree/" + name + '.png')
    #graph.write_png("E:/Trinity/Data Analytics/" + name + '.png')
    Image(graph.create_png())
    #return graph


#xl_file = pd.ExcelFile("E:/Trinity/Data Analytics/Week4/Project Data.xlsx")
#
#training_data = {sheet_name: xl_file.parse(sheet_name) 
#         for sheet_name in xl_file.sheet_names}
#
#xl_file.sheet_names
#df = xl_file.parse('Project Data')

df = pd.read_excel("E:/Trinity/Data Analytics/Assignment 2 - Decision Tree/Project Data.xlsx")

#df.shape

#Replacing the null values of X by median
df["X1"].fillna(df.X1.median(),inplace = True)
df["X2"].fillna(df.X2.median(),inplace = True)
df["X3"].fillna(df.X3.median(),inplace = True)
df["X5"].fillna(df.X5.median(),inplace = True)
df["X6"].fillna(df.X6.median(),inplace = True)
df["X7"].fillna(df.X7.median(),inplace = True)


#Replacing the null values of Y by mode
df["Y1"].fillna(df.Y1.mode()[0],inplace = True)
df["Y2"].fillna(df.Y2.mode()[0],inplace = True)
df["Y3"].fillna(df.Y3.mode()[0],inplace = True)
df["Y5"].fillna(df.Y5.mode()[0],inplace = True)
df["Y6"].fillna(df.Y6.mode()[0],inplace = True)
df["Y7"].fillna(df.Y7.mode()[0],inplace = True)


df['Y1'] = df['Y1'].astype('category')
df['Y2'] = df['Y2'].astype('category')
df['Y3'] = df['Y3'].astype('category')
df['Y4'] = df['Y4'].astype('category')
df['Y5'] = df['Y5'].astype('category')
df['Y6'] = df['Y6'].astype('category')
df['Y7'] = df['Y7'].astype('category')


#df.head()
X01 = df[['Group','X1','X2','X3','X4','X5','X6','X7','Response']]
#X01.head()
Y01 = df[['Group','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Response']]
#Y01.head()
X0 = X01[X01.Group == 0]
#X0.head()
X1 = X01[X01.Group == 1]
#X1.head()
Y0 = Y01[Y01.Group == 0]
#Y0.head()
Y1 = Y01[Y01.Group == 1]
#Y1.head()

XY01 = df.drop("ID", axis=1)
XY0 = XY01[XY01.Group == 0]
#XY0.head()
XY1 = XY01[XY01.Group == 1]
#XY1.head()


''' 
###############################################
Dataframes available
 
 1. df
 2. df_group0
 3. df_group1
 4. df_onlyX
 5. df_onlyX_group0
 6. df_onlyX_group1
 7. df_onlyY
 8. df_onlyY_group0
 9. df_onlyY_group1
 
###############################################
'''


# X01
X = X01.values[:, 0:8]
Y = X01.values[:, 8:]
graph = create_DT(X,Y,'X01')


# X0
X = X0.values[:, 0:8]
Y = X0.values[:, 8:]
graph = create_DT(X,Y,'X0')


# X1
X = X1.values[:, 0:8]
Y = X1.values[:, 8:]
graph = create_DT(X,Y,'X1')


# Y01
X = Y01.values[:, 0:8]
Y = Y01.values[:, 8:]
graph = create_DT(X.astype(int),Y.astype(int),'Y01')


# Y0
X = Y0.values[:, 0:8]
Y = Y0.values[:, 8:]
graph = create_DT(X.astype(int),Y.astype(int),'Y0')


# Y1
X = Y1.values[:, 0:8]
Y = Y1.values[:, 8:]
graph = create_DT(X.astype(int),Y.astype(int),'Y1')


# XY01
X = XY01.values[:, 1:16]
Y = XY01.values[:, 0]
graph = create_DT(X.astype(int),Y.astype(int),'XY01')


# XY0
X = XY0.values[:, 1:16]
Y = XY0.values[:, 0]
graph = create_DT(X.astype(int),Y.astype(int),'XY0')


# XY1
X = XY1.values[:, 1:16]
Y = XY1.values[:, 0]
graph = create_DT(X.astype(int),Y.astype(int),'XY1')



