import pandas as pd
from sklearn import datasets, linear_model, metrics, preprocessing, decomposition, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)


def knn_model(df,k):
    def label_endocing(col_name):
        label_encoder = preprocessing.LabelEncoder()
        df[col_name]= label_encoder.fit_transform(df[col_name])
        df[col_name].unique()

    categorical_col=[col for col in df if df[col].dtype=="object"]
    for col in categorical_col:
        label_endocing(col)
    scaler = StandardScaler()
    scaler.fit(df.drop('churn',axis = 1))
    scaled_features = scaler.transform(df.drop('churn',axis = 1))
    feature_df = pd.DataFrame(scaled_features,columns = df.columns[:-1])
    X = feature_df
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)*100
    cmat = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cmat, columns = ['Not Churn', 'Churn'], index=['Not Churn', 'Churn'])
    return accuracy, cm_df

def svm_model(df, reg):

    def label_endocing(col_name):
        label_encoder = preprocessing.LabelEncoder()
        df[col_name]= label_encoder.fit_transform(df[col_name])
        df[col_name].unique()

    categorical_col=[col for col in df if df[col].dtype=="object"]
    for col in categorical_col:
        label_endocing(col)
    scaler = StandardScaler()
    scaler.fit(df.drop('churn',axis = 1))
    scaled_features = scaler.transform(df.drop('churn',axis = 1))
    feature_df = pd.DataFrame(scaled_features,columns = df.columns[:-1])
    X = feature_df
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    clf = svm.SVC(kernel='rbf', C=reg)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cmat = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cmat, columns = ['Not Churn', 'Churn'], index=['Not Churn', 'Churn'])

    return accuracy, cm_df



