import pickle
import sklearn
def predict_iris(sepal_l,sepal_w,petal_l,petal_w,model):
    if model == "dt":
        with open("model_dt.pkl", "rb") as f:
            model = pickle.load(f)
            prediction=model.predict([[sepal_l,sepal_w,petal_l,petal_w]])[0]
            name = ['Iris setosa', 'Iris virginica', 'Iris versicolor']
            return name[prediction]
    else:
        with open("model_kn.pkl", "rb") as f:
            model = pickle.load(f)
            prediction=model.predict([[sepal_l,sepal_w,petal_l,petal_w]])[0]
            name = ['Iris setosa', 'Iris virginica', 'Iris versicolor']
            return name[prediction]