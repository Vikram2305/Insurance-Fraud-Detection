<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>Insurance Fraud Detection Web Application</h1>
    <p>This web application is designed for insurance fraud detection using machine learning techniques. It allows users
        to input details about insured individuals, vehicles, and accident information to predict whether a claim is
        genuine or potentially fraudulent.</p>
    <h2>Technologies Used</h2>
    <ul>
        <li>Python</li>
        <li>Streamlit</li>
        <li>Scikit-learn</li>
        <li>Pandas</li>
        <li>NumPy</li>
    </ul>
    <h2>Installation</h2>
    <ol>
        <li>Clone this repository:
            <code>git clone https://github.com/Vikram2305/Insurance-Fraud-Detection.git</code></li>
        <li>Navigate to the project directory:
            <code>cd Insurance-Fraud-Detection</code></li>
        <li>Install the required dependencies:
            <code>pip install -r requirements.txt</code></li>
    </ol>
    <h2>Usage</h2>
    <ol>
        <li>Run the Streamlit app:
            <code>streamlit run app.py</code></li>
        <li>Access the web application through the provided local URL.</li>
        <li>Enter the details for insured individuals, vehicles, and accident information.</li>
        <li>Click on the "Make Predictions" button to see the model's prediction regarding the claim's authenticity.</li>
    </ol>
    <h2>Model Details</h2>
    <ul>
        <li>The model used for prediction is a Random Forest Classifier trained on labeled insurance data.</li>
        <li>Data preprocessing techniques such as standard scaling and label encoding were applied before training the
            model.</li>
        <li>Model accuracy and performance metrics are detailed in the model evaluation section of the codebase.</li>
    </ul>
    <h2>File Structure</h2>
    <ul>
        <li><code>app.py</code>: Contains the Streamlit web application code for user interaction and prediction.</li>
        <li><code>fraud_detection_rf_model.pkl</code>: Pickled file containing the trained Random Forest model.</li>
        <li><code>scaler.pkl</code>: Pickled file containing the scaler used for data preprocessing.</li>
        <li><code>requirements.txt</code>: List of Python dependencies required for running the application.</li>
    </ul>
    <h2>Acknowledgments</h2>
    <p>This project is inspired by the need for efficient fraud detection mechanisms in the insurance industry. The
        codebase is based on best practices in machine learning model deployment and interactive web application
        development.</p>
</body>

</html>
