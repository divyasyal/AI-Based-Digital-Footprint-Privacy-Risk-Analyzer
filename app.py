from flask import Flask, render_template, request, redirect, url_for, session
import csv
import os
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------- LOAD DATASET ----------
csv_path = os.path.join(os.path.dirname(__file__), 'users.csv')

data = pd.read_csv(csv_path)

# Features
X = data[["devices","photos_posted","documents","metadata_leaked"]]

# ---------- CREATE RISK LABELS FOR TRAINING ----------
def label_risk(row):

    score = row["documents"] + row["metadata_leaked"]

    if score < 15:
        return "SAFE"
    elif score < 40:
        return "MODERATE"
    else:
        return "HIGH"

data["risk_level"] = data.apply(label_risk, axis=1)

y = data["risk_level"]

# ---------- TRAIN MODELS ----------

# Isolation Forest (Anomaly Detection)
anomaly_model = IsolationForest(contamination=0.2, random_state=42)
anomaly_model.fit(X)

# Random Forest (Risk Prediction)
risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X, y)


# ---------- HOME ----------
@app.route('/')
def home():
    return render_template('index.html')


# ---------- LOGIN ----------
@app.route('/login', methods=['GET','POST'])
def login_page():

    if request.method == 'POST':

        username = request.form.get('username')
        password = request.form.get('password')

        with open(csv_path, newline='') as file:
            reader = csv.DictReader(file)

            for row in reader:

                if row['username'] == username and row['password'] == password:

                    session['user'] = row
                    return redirect(url_for('dashboard'))

        return render_template('login.html', error="Invalid login")

    return render_template('login.html')


# ---------- DASHBOARD ----------
@app.route('/dashboard')
def dashboard():

    user = session.get('user')

    if not user:
        return redirect(url_for('login_page'))

    # Convert CSV values
    photos = int(user.get("photos_posted",0))
    devices = int(user.get("devices",0))
    metadata = int(user.get("metadata_leaked",0))
    documents = int(user.get("documents",0))

    # ---------- BAR CHART ----------
    bar_labels = ["Photos","Devices","Metadata","Documents"]
    bar_values = [photos,devices,metadata,documents]

    # ---------- RANDOM FOREST RISK PREDICTION ----------
    risk_prediction = risk_model.predict([[devices,photos,documents,metadata]])[0]

    probabilities = risk_model.predict_proba([[devices,photos,documents,metadata]])[0]
    confidence = round(max(probabilities)*100,2)

    risk_status = f"AI Predicted Risk: {risk_prediction} ({confidence}% confidence)"

    # ---------- PIE CHART ----------
    risk_score = metadata + documents

    safe = max(0,100-risk_score)
    moderate = metadata
    high = documents

    pie_labels = ["Safe","Moderate","High Risk"]
    pie_values = [safe,moderate,high]

    # ---------- AI RISK DOUGHNUT ----------
    risk_labels = ["Risk","Safe"]
    risk_values = [risk_score,safe]

    # ---------- ISOLATION FOREST ANOMALY DETECTION ----------
    prediction = anomaly_model.predict([[devices,photos,documents,metadata]])

    ai_alert = None
    recommendations = []

    if prediction[0] == -1:

        ai_alert = "AI detected unusual digital footprint behaviour"

        if metadata > 30:
            recommendations.append("Remove GPS metadata from photos")

        if documents > 15:
            recommendations.append("Avoid uploading sensitive documents")

        if devices > 4:
            recommendations.append("Reduce number of logged devices")

        if photos > 150:
            recommendations.append("Limit public photo sharing")

    recommendation_text = ", ".join(recommendations)

    # ---------- ANOMALY GRAPH ----------
    if prediction[0] == -1:
        anomaly_labels = ["Anomaly Detected","Normal"]
        anomaly_values = [1,0]
    else:
        anomaly_labels = ["Anomaly Detected","Normal"]
        anomaly_values = [0,1]

    # ---------- RENDER DASHBOARD ----------
    return render_template(
        "dashboard.html",
        user=user,
        bar_labels=bar_labels,
        bar_values=bar_values,
        pie_labels=pie_labels,
        pie_values=pie_values,
        risk_labels=risk_labels,
        risk_values=risk_values,
        risk_status=risk_status,
        ai_alert=ai_alert,
        recommendation_text=recommendation_text,
        anomaly_labels=anomaly_labels,
        anomaly_values=anomaly_values
    )


# ---------- LOGOUT ----------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))


# ---------- RUN APP ----------
if __name__ == "__main__":
    app.run(debug=True)