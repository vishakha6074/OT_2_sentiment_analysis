from flask import Flask, render_template, request ,jsonify
from test   import TextToNum
import pickle
app=Flask(__name__)

@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/predict",methods=["GET","POST"])

def predict():
    if request.method == "POST":
        msg = request.form.get("message")
        print(msg)

        cl=TextToNum(msg)
        cl.cleaner()
        cl.token()
        cl.removeStop()
        st=cl.stemme()
        stvc=''.join(st)
        with open("cv.pkl","rb") as vc_file:
            cv=pickle.load(vc_file)
        dt=cv.transform([stvc]).toarray()

        with open("stock_model.pkl","rb") as mb_file:
            stock_model=pickle.load(mb_file)
        pred=stock_model.predict(dt)
        print(pred)
        return render_template("result.html", prediction=pred[0]) 

    else:
        return  render_template("predict.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5050)
    