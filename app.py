from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Load pre-trained model and scaler
model = joblib.load('/home/samabishekraj/Desktop/heartproj/model/heart_disease_model.pkl')
scaler = joblib.load('/home/samabishekraj/Desktop/heartproj/model/scaler.pkl')

app = Flask(__name__) 
app.secret_key = 'your_secret_key'  # Used for session management

# Dummy credentials (for example purposes)
valid_username = 'samabi'
valid_password = 'sam@123'

# Render home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for Index (Heart Disease Predictor Form)
@app.route('/predictor')
def predictor():
    return render_template('index.html')

# Route for User Login Page
@app.route('/user_login')
def user_login():
    return render_template('userlogin.html')

# Route for direct navigation to the input form from userlogin.html
@app.route('/direct_input_form')
def direct_input_form():
    # Optionally, you can add a condition to check if the user is logged in
    session['logged_in'] = True
    return render_template('input_form.html')

# Route for the index page (login page)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == valid_username and password == valid_password:
            # Create session for the user
            session['logged_in'] = True
            return redirect(url_for('input_form'))
        else:
            # Flash an error message if login fails
            return render_template('index.html', error="Invalid credentials. Please try again.")
    
    return render_template('index.html')

# Route for the input form page
@app.route('/input_form', methods=['GET', 'POST'])
def input_form():
    # Check if the user is logged in before allowing access to the form
    if 'logged_in' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'age': float(request.form['age']),
            'sex': int(request.form['sex']),
            'cp': int(request.form['cp']),
            'trestbps': float(request.form['trestbps']),
            'chol': float(request.form['chol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': float(request.form['thalach']),
            'exang': int(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': int(request.form['slope']),
            'ca': int(request.form['ca']),
            'thal': int(request.form['thal'])
        }
        
        # Perform the prediction
        prediction, recommendation = risk_level_prediction(list(user_input.values()))
        
        # Store the data in session for use in the report
        session['risk'] = prediction
        session['recommendation'] = recommendation
        session['user_input'] = user_input
        
        return render_template(
            'results.html',
            risk=prediction,
            recommendation=recommendation,
            user_input=user_input
        )
    
    return render_template('input_form.html')

# Function to predict the risk level
def risk_level_prediction(input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 0:
        risk = "No Risk"
        advice = "ﮩـﮩﮩ٨ـ🫀ﮩ٨ـﮩﮩ٨ـLifestyle:Maintain a healthy lifestyle. Regular checkups are advised.🥗Diet:Eat a balanced diet rich in fruits, vegetables, whole grains, and lean proteins. Avoid processed foods and excessive sugar.🏋🏻‍♂️Exercise:Engage in moderate exercise for at least 30 minutes a day, 5 days a week. Activities like walking, swimming, or cycling are great options."
        
    else:
        probabilities = model.predict_proba(input_scaled)[0]
        if probabilities[1] > 0.8:
            risk = "High Risk"
            advice = "ﮩـﮩﮩ٨ـ🫀ﮩ٨ـﮩﮩ٨ـLifestyle: Immediate medical attention is recommended. Consult a cardiologist and follow their advice strictly 🥗Diet:Follow a heart-healthy diet low in saturated fats, cholesterol, and sodium. Include foods like oats, nuts, berries, and fatty fish. 🏋🏻‍♂️ Exercise:Avoid strenuous activities until cleared by a doctor. Light activities like walking or yoga may be recommended under medical supervision."
            
        elif probabilities[1] > 0.5:
            risk = "Moderate Risk"
            advice = "ﮩـﮩﮩ٨ـ🫀ﮩ٨ـﮩﮩ٨ـLifestyle:Consider lifestyle changes like stress management, quitting smoking, and limiting alcohol consumption. 🥗Diet:Reduce intake of red meat, fried foods, and sugary beverages. Focus on plant-based foods and healthy fats like olive oil 🏋🏻‍♂️ Exercise:Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week. Include strength training twice a week"
        else:
            risk = "Low Risk"
            advice = "ﮩـﮩﮩ٨ـ🫀ﮩ٨ـﮩﮩ٨ـLifestyle:Stay active and maintain a balanced lifestyle. Regular monitoring is advised 🥗Diet:Include more fiber-rich foods like beans, lentils, and whole grains. Limit salt and sugar intake. 🏋🏻‍♂️Exercise:Engage in regular physical activity like brisk walking, jogging, or dancing for at least 30 minutes most days of the week."
    
    return risk, advice

# Route for the results page
@app.route('/results')
def results():
    return render_template('results.html')

# Route for Feedback Form
@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

# Route for the Data Visualizations page
@app.route('/visualizations')
def visualizations():
    return render_template('visuals.html')

# Route for the Breathe page
@app.route('/b')
def breath():
    return render_template('bre.html')

# Logout route to end the session
@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Remove the 'logged_in' session
    return redirect(url_for('index'))

@app.route('/final_report', methods=['GET'])
def final_report():
    # Retrieve the live data from the session
    risk = session.get('risk')
    recommendation = session.get('recommendation')
    user_input = session.get('user_input')

    # Ensure valid session data exists
    if not (risk and recommendation and user_input):
        flash('No report data found. Please complete the prediction first.')
        return redirect(url_for('input_form'))
    
    return render_template(
        'final_report.html',
        risk=risk,
        recommendation=recommendation,
        user_input=user_input
    )

if __name__ == '__main__':
    app.run(debug=True)