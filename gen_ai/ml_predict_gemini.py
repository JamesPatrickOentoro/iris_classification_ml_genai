import google.generativeai as genai
import os

genai.configure(api_key='')
model = genai.GenerativeModel("gemini-1.5-flash")

def predict_iris_gen(sepal_l,sepal_w,petal_l,petal_w):
    response = model.generate_content(f"According to the iris dataset. What is the Iris species if it has sepal lenght = {sepal_l}, sepal width = {sepal_w}, petal lenght = {petal_l}, petal width = {petal_w}. Answer with the species name only ['Iris setosa', 'Iris virginica', 'Iris versicolor']")
    return response.text

print(predict_iris_gen(5,5,5,5))