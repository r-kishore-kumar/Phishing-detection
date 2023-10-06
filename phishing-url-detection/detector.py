import numpy as np
import pandas as pd
from flask import Flask,render_template,request
from sklearn.ensemble import RandomForestClassifier
from url_processing import FeatureExtraction

app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def validate_url():
    if request.method=="POST":
        Url=request.form.get('url')

        obj = FeatureExtraction(Url)
        Xtest=obj.getFeaturesList()

        X_test = np.array([Xtest])

        val=rf_classifier.predict(X_test)
        if val==-1:
            return f'<h1>{Url} is NOT SAFE to Use!!!</h1>'

        return f'<h1>{Url} is Validated! SAFE to Use</h1>'
    return render_template('index.html')




if __name__ == '__main__':

    # Load the dataset from a CSV file
    df = pd.read_csv("phishing.csv")
    X = df.drop(["class","Index"], axis=1) # features
    y = df["class"]              # target

    # model creation
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X,y)

    app.run(debug=True)

