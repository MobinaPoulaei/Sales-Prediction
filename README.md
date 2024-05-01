# Flask Web Application for Sales Prediction

This repository also contains a Flask web application (`app.py`) that provides a simple web interface for making sales predictions using the trained Random Forest model stored in `random_forest_model.pkl`.

## Overview

The Flask application allows users to input features through a web form, which then uses the pre-trained model to predict and display sales data. It serves an HTML page for user interaction.

## Files

- `app.py`: The Flask application script.
- `index.html`: The HTML template for the web interface (not provided, assumed to be in the same directory).
- `random_forest_model.pkl`: The trained Random Forest model used for making predictions.

## Setup

### Prerequisites

- Python 3.8 or newer
- Flask
- pandas
- numpy
- scikit-learn

Ensure you have the above Python libraries installed, along with Flask. You can install Flask using pip:

```bash
pip install Flask
```

### Running the Flask Application

1. **Prepare Your Environment:**
   - Ensure that `app.py`, `index.html`, and `random_forest_model.pkl` are in the same directory.
   - If `index.html` is not present, you will need to create this file with the appropriate HTML form elements and placeholders for displaying predictions.

2. **Start the Application:**
   - Navigate to the directory containing `app.py`.
   - Run the Flask application using the following command:
     ```bash
     python app.py
     ```
   - This will start a local web server.

3. **Access the Webpage:**
   - Open a web browser and go to `http://127.0.0.1:3000/`.
   - Use the form to enter the necessary input values and submit them to predict sales.

## Functionality

- The Flask app defines two routes:
  - `/`: Serves the main page where users can input their data.
  - `/predict`: Handles the POST request from the form, performs the prediction using the loaded Random Forest model, and returns the result to the same page.

## Deployment

To make this application accessible over the Internet, consider deploying it to a cloud platform like Heroku, AWS, or Google Cloud Platform. Ensure you manage your environment variables and dependencies correctly for production environments.
