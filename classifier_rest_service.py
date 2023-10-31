from flask import Flask,request

import pickle

import numpy as np
local_classifier = pickle.load(open('classifier.pickle','rb'))
local_scalar = pickle.load(open('sc.pickle','rb'))

# new_pred_prob = local_classifier.predict_proba(local_scalar.transform(np.array([[40,20000]])))

app=Flask(__name__)
@app.route('/model',methods=['POST'])
def hello_world():
    request_data=request.get_json(force=True)
    age = request_data['age']
    salary = request_data['salary']
    print(age)
    print(salary)
    new_pred = local_classifier.predict_proba(local_scalar.transform(np.array([[age,salary]])))

    return "The Prediction is {[0]} ".format(new_pred)
if __name__=="__main__":
    app.run(port=8002,debug=True)
    
