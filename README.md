## CDR Fraud Detection with FastAPI

### Overview
This project is a complete deployment-ready **CDR (Call Detail Record) Fraud Detection** web application.
It utilizes a pre-trained Random Forest model to predict potential telecom fraud based on user input.

The project includes:
- A RESTful API built with **FastAPI**
- An interactive and responsive web UI
- Docker containerization for easy deployment
- Deployable on **Render** with auto-deploy support

---

### Features
- `/predict` endpoint for fraud detection predictions
- `/health` endpoint for status checking
- `/ui` endpoint for an intuitive web interface
- Fraud threshold adjustable via API
- Fully dockerized and tested deployment

---

### 📂 Project Structure
```
.
├── app.py                  # FastAPI backend application
├── templates/
│   └── index.html          # Frontend web UI
├── static/                 # Static assets (if any)
├── random_forest_model.pkl # Trained Random Forest model
├── Dockerfile              # Docker setup
├── render.yaml             # Render deployment configuration
├── requirements.txt        # Project dependencies
└── .gitignore
```

---

### Deployment
The app is deployed live on Render and can be accessed here:  
👉 **[Live Demo on Render](https://cdr-fraud-detection-fastapi.onrender.com/ui)**

---

### Tech Stack
- **FastAPI** for backend API
- **scikit-learn** for machine learning model
- **Pandas & Numpy** for data processing
- **Docker** for containerization
- **Render** for deployment
- **Tailwind CSS** for UI styling

---

### How to Run Locally
```bash
git clone https://github.com/Arsney091289421/cdr-fraud-detection-fastapi.git
cd cdr-fraud-detection-fastapi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Visit `http://localhost:8000/ui` in your browser.

---

### Model Information
- The model used is a RandomForestClassifier trained on a public CDR dataset.
- Threshold optimization was done to balance precision and recall.
- Default threshold set to **0.46** based on evaluation.

---

### License
This project is licensed under the [MIT License](LICENSE).

---

### Author
**Daniel (Arsney091289421)**

