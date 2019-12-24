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
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)

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
    graph.write_png("E:/Trinity/Data Analytics/Assignment 2 - Decision Tree/NoImpute" + name + '.png')
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

# Drop all null values 
df = df.dropna()


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
 
 1. XY01
 2. XY0
 3. XY1
 4. X01
 5. X0
 6. X1
 7. Y01
 8. Y0
 9. Y1
 
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



