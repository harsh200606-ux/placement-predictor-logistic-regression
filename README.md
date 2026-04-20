# Student Placement ML Website

A full working machine learning website using:

- Flask
- scikit-learn
- Logistic Regression
- HTML/CSS/JavaScript

## Project files

- `app.py` -> Flask backend
- `placement_model.pkl` -> trained real ML model
- `student_placement_data.csv` -> dataset used for training
- `templates/index.html` -> frontend
- `static/style.css` -> styles

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open the local address shown in the terminal.

## Model info

This project uses a real Logistic Regression model trained on the included dataset.

Test accuracy on held-out data: 91.00%

## Input fields

- CGPA
- Aptitude Score
- Communication Score
- Internship
- Projects
- Backlogs
- Certifications
