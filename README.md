# 🏥 PatientConnect App

A comprehensive, user-friendly healthcare portal built with **Streamlit** and **MongoDB**, designed to streamline patient and administrator interactions — from booking appointments to managing medical records, billing, and AI-powered support.

> "One app to connect patients, doctors, and healthcare services — with intelligence."

---

## 🚀 Features

### 🩺 Patient Portal

* **Doctor Discovery & Favorites**

  * Search doctors by name or specialty
  * Add to favorites & initiate secure messaging
  * WhatsApp integration for direct contact
* **Appointment Management**

  * Schedule, view, and manage appointments
  * Real-time status updates (Scheduled/Cancelled)
* **Medical Records Hub**

  * Upload prescriptions, lab reports, images
  * AI-generated summaries of uploaded records
  * Preview and manage documents
* **Billing & Payments**

  * View billing history with status tracking (Due/Paid)
  * Simple UI for payment follow-ups
* **Insurance Claims**

  * Track and add insurance claim records
* **Secure Messaging**

  * Chat with doctors via secure messages
  * Inbox with read/unread tracking
* **Support Tickets**

  * Raise issues, feedback, and view responses
* **Emergency & Location Services**

  * Quick access to emergency contacts
  * AI-assisted nearby hospital finder

### 🛠️ Admin Portal

* **Dashboard & Metrics**

  * KPIs for revenue, appointments, tickets, patients
  * Doctor-wise performance charts (Revenue, Appointments)
* **Doctor Management**

  * Add/Delete doctor profiles with avatar generation
* **Patient Management**

  * Add and manage patient credentials and profiles
* **Billing Administration**

  * Add bills, update statuses, track payments
* **Support Ticket Handling**

  * View, respond, and update patient tickets with status control

### 🤖 AI Features (Pluggable LLM Service)

* LLM-powered chatbot for patient support
* AI-generated summaries of medical records & files
* Location-based services using AI context queries
* Voice output via browser SpeechSynthesis

---

## 🧑‍💻 Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Backend Database**: [MongoDB](https://www.mongodb.com/)
* **AI Services**: External LLM service (`llm_service.py`)
* **Security**: [passlib bcrypt](https://passlib.readthedocs.io/en/stable/)
* **Visualization**: Matplotlib, Pandas

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/patientconnect.git
cd patientconnect
pip install -r requirements.txt
```

Ensure MongoDB is running locally or update the `MONGODB_URI` environment variable for production use.

---

## 🏃 Running the App

```bash
streamlit run patientconnect.py
```

* The app will launch at `http://localhost:8501`.
* Admin hardcoded credentials: `admin / adminpass` (for demo purposes).
* Patient login is based on MongoDB user collection.

---

## 🗂 Project Structure

```
📦 patientconnect/
 ┣ 📄 patientconnect.py      # Main Streamlit Application
 ┣ 📄 llm_service.py         # AI integration functions (pluggable)
 ┣ 📄 requirements.txt       # Python dependencies
 ┗ 📁 /tmp/patientconnect/   # Temporary uploads directory
```

---

## 🔐 Environment Variables

* `MONGODB_URI` — Connection string for MongoDB (default: localhost)

---

## 📊 Sample Data Schema

### Users (Patients & Admins)

```json
{
  "username": "john_doe",
  "hashed_password": "bcrypt-hash",
  "role": "patient",
  "patient_id": "p123",
  "profile": { "name": "John Doe", "age": 30, "gender": "Male" }
}
```

### Doctors

```json
{
  "id": "d001",
  "name": "Dr. Jane Smith",
  "specialty": "Cardiologist",
  "rating": 4.5,
  "photo_url": "https://api.dicebear.com/7.x/avataaars/svg?seed=JaneSmith",
  "email": "dr.jane@example.com",
  "phone": "+91-9876543210"
}
```

---

## 📝 TODOs & Roadmap

* [ ] Role-based access control (RBAC)
* [ ] Integration with real payment gateways
* [ ] Notification services (SMS, WhatsApp APIs)
* [ ] OAuth login & JWT session handling
* [ ] Production deployment with Docker & Nginx
* [ ] Audit logs for admin actions

---

## ⚠️ Disclaimer

This application is a **demo project** for learning and prototype purposes. Do not use in production without security hardening, authentication improvements, and compliance checks.

---

## 💙 Contributing

PRs are welcome! Feel free to fork & enhance the PatientConnect App.


