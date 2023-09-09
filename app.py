from flask import Flask, render_template, request
import iris_flower_classification_copy 

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def basic():
    if request.method == "POST":
        sepal_length = request.form['sepallength']
        petal_length = request.form['petallength']
        sepal_width = request.form['sepalwidth']
        petal_width = request.form['petalwidth']
        y_pred = [[sepal_length, sepal_width, petal_length, petal_width]]
        trained_model = iris_flower_classification_copy.training_model()
        prediction_value = trained_model.predict(y_pred)

        setosa = 'setosa'
        versicolor = 'versicolor'
        virginica = 'virginica'
        if prediction_value == 0:
            return render_template('index.html', setosa = setosa)
        elif prediction_value == 1:
            return render_template('index.html', versicolor = versicolor)
        else:
            return render_template('index.html', virginica=virginica)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

