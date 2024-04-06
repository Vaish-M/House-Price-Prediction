from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
app=Flask(__name__)
df=pd.read_csv('data.csv')
X=df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']]
y=df['price']
X_train, X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train, y_train)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    bedrooms=float(request.form['bedrooms'])
    bathrooms=float(request.form['bathrooms'])
    sqft_living=float(request.form['sqft_living'])
    sqft_lot=float(request.form['sqft_lot'])
    floors=float(request.form['floors'])
    waterfront=int(request.form['waterfront'])
    view=int(request.form['view'])
    condition=int(request.form['condition'])
    new_data=[[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition]]
    predicted_price=int(model.predict(new_data)[0])
    return render_template('index.html', predicted_price=predicted_price)
if __name__ == '__main__':
    app.run(debug=True)
