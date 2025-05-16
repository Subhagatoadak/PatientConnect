# -*- coding: utf-8 -*-
"""
PatientConnect Streamlit Application with MongoDB Integration.

This application provides a portal for patients and administrators to manage
healthcare information, including appointments, records, billing, and support,
with data stored in MongoDB and AI features provided by an external service.
"""

# Standard Library Imports
import os
import uuid
import base64
import json
import io
import re
from datetime import datetime, date, time, timedelta

# Third-Party Imports
import streamlit as st
from streamlit.components.v1 import html as st_html
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import pymongo
from passlib.context import CryptContext
from bson.objectid import ObjectId

# (Optional: Only if fetching images from external URLs directly)
# import requests

# Local Application/Service Imports (Ensure llm_service.py exists)
try:
    from llm_service import (
        generate_llm_response,
        generate_image_description,
        generate_llm_json
    )
except ImportError:
    st.error(
        "CRITICAL: llm_service module not found. AI features will fail. "
        "Please ensure llm_service.py exists or the package is installed."
    )
    # Define basic placeholders if the import fails

    def generate_llm_response(prompt: str, context: dict = None) -> str:
        """Placeholder LLM response function."""
        print(f"WARNING: Placeholder LLM. Prompt: {prompt}")
        return f"Placeholder response (no llm_service): {prompt}"

    def generate_image_description(img_path: str, prompt: str) -> str:
        """Placeholder image description function."""
        # Note: This placeholder differs from the user-provided one,
        # as it doesn't need the encode_image helper.
        print(f"WARNING: Placeholder Image Desc. Prompt: {prompt}, Path: {img_path}") # noqa E501
        return "Placeholder summary (no llm_service)."

    def generate_llm_json(prompt: str, context: dict = None) -> dict:
        """Placeholder LLM JSON function."""
        print(f"WARNING: Placeholder LLM JSON. Prompt: {prompt}")
        return {"error": "no llm_service", "prompt": prompt}


# ---- Constants ----
# Using environment variable is strongly recommended for production URIs
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "patient_connect_db"
# DO NOT USE THIS IN PRODUCTION - store securely or remove if using DB only
ADMIN_CREDENTIALS = {"admin": "adminpass"}
MAX_FILE_UPLOAD_MB = 10
INDIA_COUNTRY_CODE = "+91"
TEMP_DIR = "/tmp/patientconnect"  # Temporary directory for file uploads

# Ensure temp directory exists
if not os.path.exists(TEMP_DIR):
    try:
        os.makedirs(TEMP_DIR)
        print(f"Created temporary directory: {TEMP_DIR}")
    except OSError as e:
        st.error(f"Fatal: Could not create temporary directory: {TEMP_DIR}. Error: {e}") # noqa E501
        # Stop execution if temp dir is critical and cannot be created
        st.stop()


# ---- Security Setup ----
# Using passlib for password hashing (requires: pip install passlib[bcrypt])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---- Page Configuration & Global CSS ----
st.set_page_config(
    page_title="PatientConnect", layout="wide", initial_sidebar_state="auto"
)
# ---- Database Connection ----
@st.cache_resource
def get_mongo_client():
    """
    Establishes and caches a MongoDB client connection.

    Returns:
        pymongo.MongoClient: The MongoDB client instance, or None if connection fails. # noqa E501
    """
    try:
        client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000) # noqa E501
        # Check connection
        client.admin.command('ping')
        print("MongoDB connection successful.")
        return client
    except pymongo.errors.ConnectionFailure as e:
        st.error(f"Fatal: Could not connect to MongoDB at {MONGODB_URI}. Check URI and server status. Error: {e}") # noqa E501
        print(f"MongoDB Connection Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during MongoDB connection: {e}") # noqa E501
        print(f"Unexpected MongoDB Connection Error: {e}")
        return None


def get_db():
    """
    Gets the specific database instance from the cached client connection.

    Returns:
        pymongo.database.Database: The database instance, or None if client is unavailable. # noqa E501
    """
    client = get_mongo_client()
    if client:
        try:
            return client[DATABASE_NAME]
        except Exception as e:
            st.error(f"Error accessing database '{DATABASE_NAME}': {e}")
            return None
    return None


# Establish DB connection; stop app if connection fails
DB = get_db()
if DB is None:
    st.error("Application cannot start without a database connection.")
    st.stop()

# Define collection variables for easier access
USERS_COLL = DB["users"]
DOCTORS_COLL = DB["doctors"]
APPOINTMENTS_COLL = DB["appointments"]
RECORDS_COLL = DB["medical_records"]
BILLING_COLL = DB["billing"]
TICKETS_COLL = DB["tickets"]
MESSAGES_COLL = DB["messages"]
CLAIMS_COLL = DB["insurance_claims"]
FAVORITES_COLL = DB["favorites"]


# ---- Password Hashing Utilities ----
def verify_password(plain_password, hashed_password):
    """Verifies a plain password against a hashed password."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        # Log error, potentially related to hash format
        print(f"Error verifying password: {e}")
        return False


def hash_password(password):
    """Hashes a plain password using bcrypt."""
    return pwd_context.hash(password)


# ---- Image Encoding Helper ----
def encode_image(image_path):
    """Reads an image file and encodes it as a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        st.error(f"Error: Image file not found at path: {image_path}")
        return None
    except Exception as e:
        st.error(f"Error encoding image file {image_path}: {e}")
        return None




# Global CSS (Ensure proper indentation)
st.markdown(
    r"""
    <style>
        /* General */
        .stApp { }

        /* Header */
        .header {
            background-color: #007bff; /* Professional Blue */
            padding: 15px 10px; border-radius: 5px; font-size: 36px;
            color: white; font-weight: bold; text-align: center;
            margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Cards */
        .profile-card, .info-card {
            border: 1px solid #ddd; border-radius: 8px; padding: 20px;
            background-color: #ffffff; margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .stDataFrame { width: 100% !important; } /* Style dataframes */

        /* Buttons */
        .stButton>button {
            border-radius: 20px; border: 1px solid #007bff; margin: 2px;
            background-color: #007bff; color: white; padding: 8px 16px;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3; color: white; border: 1px solid #0056b3;
        }
        .stButton>button[kind="secondary"] { /* Secondary buttons */
             background-color: #6c757d; border-color: #6c757d;
        }
       .stButton>button[kind="secondary"]:hover {
             background-color: #5a6268; border-color: #545b62;
       }
       .stButton>button.delete-button { /* Custom class for delete */
             background-color: #dc3545; border-color: #dc3545; color: white;
        }
       .stButton>button.delete-button:hover {
            background-color: #c82333; border-color: #bd2130;
       }

        /* Floating Chat (Only for Patient App) */
        .float-chat-container {
            position: fixed; bottom: 20px; right: 20px; width: 350px;
            z-index: 1000; background: #ffffff; border: 1px solid #ccc;
            border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .float-chat-container .stExpander { /* Style expander */
            border: none !important; margin: 0; padding: 0;
        }
        .float-chat-container .stExpander header { /* Expander header */
            background-color: #007bff; color: white; font-weight: bold;
            border-top-left-radius: 10px; border-top-right-radius: 10px;
            padding: 10px 15px !important; /* Use !important carefully */
        }
        .float-chat-container .stExpander [data-testid="stExpanderDetails"] {
            max-height: 400px; overflow-y: auto; padding: 10px;
            background-color: #f9f9f9; /* Light background for chat area */
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Display Header Globally
st.markdown('<div class="header">PatientConnect</div>', unsafe_allow_html=True)


# ---- Session State Initialization ----
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    defaults_state = {
        "patient_authenticated": False,
        "admin_authenticated": False,
        "current_user_db_id": None,
        "current_patient_logical_id": None,
        "patient_profile": {},
        "doctors": [], # Loaded from DB on demand or login
        "favorites": set(),
        "appointments": [],
        "medical_records": [],
        "billing_history": [],
        "tickets": [],
        "messages": [],
        "insurance_claims": [],
        "chat_history": [
            {"role": "assistant", "content": "Hi there! How can I help?"}
        ],
        "session_initialized": True
    }
    for key, value in defaults_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize state once per session
if 'session_initialized' not in st.session_state:
    initialize_session_state()


# ---- Global Helper Functions ----
def load_user_data(user_doc):
    """Loads data for the logged-in patient from MongoDB into session state."""
    if not user_doc or user_doc.get("role") != "patient":
        st.error("Invalid user data for patient login.")
        logout()
        return

    user_db_id = user_doc["_id"]
    # Use logical ID if available, fallback to string of ObjectId
    patient_logical_id = user_doc.get("patient_id", str(user_db_id))

    st.session_state.current_user_db_id = user_db_id
    st.session_state.current_patient_logical_id = patient_logical_id
    # Store profile info + username from the user document
    st.session_state.patient_profile = user_doc.get("profile", {})
    st.session_state.patient_profile["username"] = user_doc["username"]

    # Load related data, converting ObjectIds to strings where necessary
    # if they will be used as keys or need serialization later.
    try:
        # Favorites (simple lookup by user's DB ID)
        fav_docs = FAVORITES_COLL.find({"user_id": user_db_id})
        st.session_state.favorites = {fav["doctor_id"] for fav in fav_docs}

        # Other collections filtered by patient's logical ID
        appt_query = {"patient_id": patient_logical_id}
        st.session_state.appointments = list(APPOINTMENTS_COLL.find(appt_query).sort("datetime", pymongo.DESCENDING)) # noqa E501

        rec_query = {"patient_id": patient_logical_id}
        st.session_state.medical_records = list(RECORDS_COLL.find(rec_query).sort("uploaded_at", pymongo.DESCENDING)) # noqa E501

        bill_query = {"patient_id": patient_logical_id}
        st.session_state.billing_history = list(BILLING_COLL.find(bill_query).sort("date", pymongo.DESCENDING)) # noqa E501

        ticket_query = {"patient_id": patient_logical_id}
        st.session_state.tickets = list(TICKETS_COLL.find(ticket_query).sort("submitted_at", pymongo.DESCENDING)) # noqa E501

        # Messages involving the patient (using logical ID)
        msg_query = {"$or": [{"from_id": patient_logical_id},
                             {"to_id": patient_logical_id}]}
        st.session_state.messages = list(MESSAGES_COLL.find(msg_query).sort("timestamp", pymongo.DESCENDING)) # noqa E501

        claim_query = {"patient_id": patient_logical_id}
        st.session_state.insurance_claims = list(CLAIMS_COLL.find(claim_query).sort("claim_date", pymongo.DESCENDING)) # noqa E501

        # Ensure doctors list is loaded (can be optimized)
        if not st.session_state.doctors:
            st.session_state.doctors = list(DOCTORS_COLL.find())

        # Reset chat history for new login
        st.session_state.chat_history = [
            {"role": "assistant",
             "content": f"Hi {st.session_state.patient_profile.get('name', 'there')}! How can I help?"} # noqa E501
        ]
        print(f"Data loaded for user: {user_doc['username']}")

    except Exception as e:
        st.error(f"Error loading patient data from database: {e}")
        logout()  # Log out if essential data loading fails


def logout():
    """Logs out user by resetting session state authentication flags."""
    keys_to_reset = [
        "patient_authenticated", "admin_authenticated", "current_user_db_id",
        "current_patient_logical_id", "patient_profile", "favorites",
        "appointments", "medical_records", "billing_history", "tickets",
        "messages", "insurance_claims"
    ]
    # Reset specific keys to initial-like state
    for key in keys_to_reset:
        if key.endswith('_authenticated'):
            st.session_state[key] = False
        elif key == 'patient_profile':
            st.session_state[key] = {}
        elif key == 'favorites':
            st.session_state[key] = set()
        elif isinstance(st.session_state.get(key), list):
            st.session_state[key] = []
        else:
            st.session_state[key] = None

    # Reset chat history
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi there! How can I help you today?"}
    ]
    # Keep doctors list cached maybe? Optional.
    # st.session_state.doctors = []
    st.toast("Logged out successfully.", icon="👋")


def is_valid_email(email):
    """Basic email format validation."""
    if not email: return False
    # Stricter pattern based on common standards
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def get_doctor_details(doctor_id):
    """Retrieves doctor details (uses cached list or queries DB)."""
    # Check cache first (handle ObjectId or string id)
    for doc in st.session_state.get('doctors', []):
        if str(doc.get('_id')) == str(doctor_id) or doc.get('id') == str(doctor_id): # noqa E501
            return doc
    # If not in cache, query DB
    try:
        query = {}
        if ObjectId.is_valid(doctor_id):
            query = {"_id": ObjectId(doctor_id)}
        else:
            # Try querying by logical 'id' if it's used
            query = {"id": str(doctor_id)}

        doc_from_db = DOCTORS_COLL.find_one(query)
        if doc_from_db:
            # Optionally update cache? Be careful about cache size.
            # st.session_state.doctors.append(doc_from_db)
            return doc_from_db
    except Exception as e:
        print(f"Error querying doctor {doctor_id} from DB: {e}")
    return None


def get_doctor_name(doctor_id):
    """Retrieves doctor name by ID."""
    doc = get_doctor_details(doctor_id)
    return doc.get('name', "Unknown Doctor") if doc else "Unknown Doctor"


def format_phone_number_for_whatsapp(phone):
    """Cleans and formats phone number for WhatsApp link."""
    if not phone: return None
    cleaned_phone = ''.join(filter(str.isdigit, phone))
    # Basic check for India-like numbers
    if not cleaned_phone.startswith('91') and len(cleaned_phone) == 10:
        return INDIA_COUNTRY_CODE[1:] + cleaned_phone # Add 91 without +
    elif cleaned_phone.startswith('91') and len(cleaned_phone) == 12:
        return cleaned_phone # Already includes 91
    # For other formats, return cleaned digits; may need more logic
    return cleaned_phone


# --- Database Interaction Functions ---
def add_appointment_db(patient_logical_id, doctor_logical_id, dt):
    """Adds appointment document to MongoDB."""
    if not patient_logical_id or not doctor_logical_id: return None
    appointment_doc = {
        "patient_id": patient_logical_id, "doctor_id": doctor_logical_id,
        "datetime": dt, "created_at": datetime.now(), "status": "Scheduled"
    }
    try:
        result = APPOINTMENTS_COLL.insert_one(appointment_doc)
        appointment_doc["_id"] = result.inserted_id
        return appointment_doc
    except Exception as e:
        st.error(f"Database Error: Could not book appointment. {e}")
        return None


def add_record_db(patient_logical_id, record_type, filename, file_type,
                  description, summary, size_kb):
    """Adds medical record metadata to MongoDB."""
    if not patient_logical_id: return None
    # Store a simulated storage reference/path instead of file content
    storage_ref = f"simulated/{patient_logical_id}/{uuid.uuid4()}_{filename}"
    record_doc = {
        "patient_id": patient_logical_id, "type": record_type,
        "uploaded_at": datetime.now(), "filename": filename,
        "file_type": file_type, "description": description or "N/A",
        "summary": summary, "size_kb": size_kb,
        "storage_ref": storage_ref
    }
    try:
        result = RECORDS_COLL.insert_one(record_doc)
        record_doc["_id"] = result.inserted_id
        return record_doc
    except Exception as e:
        st.error(f"Database Error: Could not save record metadata. {e}")
        return None


def add_ticket_db(patient_logical_id, subject, details):
    """Adds support ticket document to MongoDB."""
    if not patient_logical_id: return None
    ticket_doc = {
        "patient_id": patient_logical_id, "subject": subject,
        "details": details, "status": "Open",
        "submitted_at": datetime.now(), "comments": []
    }
    try:
        result = TICKETS_COLL.insert_one(ticket_doc)
        ticket_doc["_id"] = result.inserted_id
        return ticket_doc
    except Exception as e:
        st.error(f"Database Error: Could not submit ticket. {e}")
        return None


def add_message_db(from_logical_id, to_logical_id, subject, body):
    """Adds message document to MongoDB."""
    message_doc = {
        "from_id": from_logical_id, "to_id": to_logical_id,
        "subject": subject, "body": body,
        "timestamp": datetime.now(), "read": False,
    }
    try:
        result = MESSAGES_COLL.insert_one(message_doc)
        message_doc["_id"] = result.inserted_id
        # Invalidate local message cache if needed, or let load_user_data handle # noqa E501
        return message_doc
    except Exception as e:
        st.error(f"Database Error: Could not send message. {e}")
        return None


def add_claim_db(patient_logical_id, claim_date, provider, amount, details):
    """Adds insurance claim document to MongoDB."""
    if not patient_logical_id: return None
    claim_doc = {
        "patient_id": patient_logical_id,
        "claim_date": datetime.combine(claim_date, time.min), # Store date
        "amount": amount, "provider": provider,
        "status": "Submitted", "details": details,
    }
    try:
        result = CLAIMS_COLL.insert_one(claim_doc)
        claim_doc["_id"] = result.inserted_id
        return claim_doc
    except Exception as e:
        st.error(f"Database Error: Could not add claim. {e}")
        return None


def add_favorite_db(user_db_id, doctor_logical_id):
    """Adds a favorite relationship, avoiding duplicates."""
    if not user_db_id or not doctor_logical_id: return False
    fav_doc = {"user_id": user_db_id, "doctor_id": doctor_logical_id}
    try:
        # Use update_one with upsert to add if not present, do nothing if present # noqa E501
        FAVORITES_COLL.update_one(
            fav_doc, {"$set": fav_doc}, upsert=True
        )
        return True
    except Exception as e:
        st.error(f"Database Error: Could not add favorite. {e}")
        return False


def remove_favorite_db(user_db_id, doctor_logical_id):
    """Removes a favorite relationship."""
    if not user_db_id or not doctor_logical_id: return False
    try:
        result = FAVORITES_COLL.delete_one(
            {"user_id": user_db_id, "doctor_id": doctor_logical_id}
        )
        return result.deleted_count > 0 # Return True if something was deleted
    except Exception as e:
        st.error(f"Database Error: Could not remove favorite. {e}")
        return False


def update_appt_status_db(appt_db_id, new_status):
    """Updates the status of an appointment in MongoDB."""
    if not ObjectId.is_valid(appt_db_id):
        st.error("Invalid appointment ID format for update.")
        return False
    try:
        result = APPOINTMENTS_COLL.update_one(
            {"_id": ObjectId(appt_db_id)},
            {"$set": {"status": new_status}}
        )
        return result.modified_count > 0
    except Exception as e:
        st.error(f"Database Error: Could not update appointment status. {e}")
        return False


def add_ticket_comment_db(ticket_db_id, actor_name, comment_text):
    """Adds a comment to a ticket's comment array in MongoDB."""
    if not ObjectId.is_valid(ticket_db_id):
        st.error("Invalid ticket ID format for adding comment.")
        return False
    new_comment = {
        "_id": ObjectId(), # Give comment its own unique ID
        "timestamp": datetime.now(),
        "actor": actor_name,
        "text": comment_text,
    }
    try:
        result = TICKETS_COLL.update_one(
            {"_id": ObjectId(ticket_db_id)},
            {"$push": {"comments": new_comment}}
        )
        return result.modified_count > 0
    except Exception as e:
        st.error(f"Database Error: Could not add ticket comment. {e}")
        return False


def update_ticket_status_db(ticket_db_id, new_status):
    """Updates the status of a ticket in MongoDB."""
    if not ObjectId.is_valid(ticket_db_id):
        st.error("Invalid ticket ID format for status update.")
        return False
    try:
        result = TICKETS_COLL.update_one(
            {"_id": ObjectId(ticket_db_id)},
            {"$set": {"status": new_status}}
        )
        return result.modified_count > 0
    except Exception as e:
        st.error(f"Database Error: Could not update ticket status. {e}")
        return False


def mark_message_read_db(msg_db_id):
    """Marks a message as read in MongoDB."""
    if not ObjectId.is_valid(msg_db_id):
        print(f"Invalid message ID format: {msg_db_id}")
        return False
    try:
        result = MESSAGES_COLL.update_one(
            {"_id": ObjectId(msg_db_id)},
            {"$set": {"read": True}}
        )
        return result.modified_count > 0
    except Exception as e:
        # Log silently for background task? Or maybe show subtle UI feedback # noqa E501
        print(f"Error marking message {msg_db_id} as read: {e}")
        return False


# ---- Floating Chatbot Renderer ----
def render_floating_chatbot():
    """Renders the floating chatbot, using imported LLM functions."""
    with st.container():
        st.markdown('<div class="float-chat-container">',
                    unsafe_allow_html=True)
        with st.expander("🤖 AI Health Assistant", expanded=False):
            # Display chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])

            # Chat input
            if prompt := st.chat_input("Ask me anything..."):
                st.session_state.chat_history.append(
                    {'role': 'user', 'content': prompt}
                )
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Prepare context, converting ObjectIds if needed
                context_for_llm = {
                    "patient_profile": st.session_state.patient_profile,
                    "appointments": [
                        {k: (str(v) if isinstance(v, ObjectId) else v)
                         for k, v in appt.items()}
                        for appt in st.session_state.appointments[-5:]
                    ],
                    "records_summary": [
                        f"{r.get('filename','N/A')} ({r.get('type','N/A')}): {r.get('summary','N/A')}" # noqa E501
                        for r in st.session_state.medical_records[-3:]
                    ],
                    "current_view": st.session_state.get("current_page",
                                                          "Unknown"),
                    "doctors": {
                        str(d.get('_id', d.get('id'))): {
                            'name': d['name'], 'specialty': d['specialty']
                        } for d in st.session_state.doctors
                    },
                    "location": {"city": "Ranchi", "state": "Jharkhand",
                                 "country": "India"}
                }

                # Use imported function directly
                try:
                    response = generate_llm_response(prompt,
                                                     context=context_for_llm)
                except Exception as e:
                    st.error(f"AI response generation failed: {e}")
                    response = ("Sorry, error generating response.")

                st.session_state.chat_history.append(
                    {'role': 'assistant', 'content': response}
                )
                with st.chat_message("assistant"):
                    st.markdown(response)

                # JS for voice output
                response_js_safe = json.dumps(response)
                js_code = f"""
                <script>
                    try {{
                        var msg = new SpeechSynthesisUtterance({response_js_safe}); // noqa E501
                        window.speechSynthesis.speak(msg);
                    }} catch (e) {{ console.error("Speech error:", e); }}
                </script>
                """
                st_html(js_code, height=0, width=0)

        st.markdown('</div>', unsafe_allow_html=True)


# ---- Patient App Renderer ----
def render_patient_app():
    """Renders the entire Patient Interface using data from session state."""
    patient_logical_id = st.session_state.current_patient_logical_id
    user_db_id = st.session_state.current_user_db_id
    if not patient_logical_id or not user_db_id:
        st.error("Error: Patient session invalid. Please login again.")
        logout(); st.rerun(); return

    # -- Patient Sidebar --
    st.sidebar.title("Patient Menu")
    st.sidebar.markdown("---")
    profile_name = st.session_state.patient_profile.get('name', 'Patient')
    st.sidebar.success(f"Logged in as: {profile_name}")
    menu_options = {
        "Dashboard": "🏠", "Doctor Management": "🧑‍⚕️", "Appointments": "📅",
        "Medical Records": "📚", "Billing & Payments": "💳",
        "Insurance Claims": "📄", "Secure Messaging": "✉️",
        "Patient Support": "📢", "Emergency & Location": "📍",
    }
    page_keys = list(menu_options.keys()); current_page_index = 0
    # Handle navigation flags set by buttons to pre-select page
    if st.session_state.get("go_to_messaging"):
        try: current_page_index = page_keys.index("Secure Messaging")
        except ValueError: pass
        st.session_state.go_to_messaging = False # Reset flag
    elif st.session_state.get("go_to_appointments"):
        try: current_page_index = page_keys.index("Appointments")
        except ValueError: pass
        st.session_state.go_to_appointments = False # Reset flag

    page = st.sidebar.radio(
        "Go to", page_keys, index=current_page_index,
        format_func=lambda x: f"{menu_options[x]} {x}", key="patient_nav"
    )
    st.session_state["current_page"] = page # Store for context

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", key="patient_logout"):
        logout(); st.rerun()

    # -- Patient Page Rendering --
    if page == "Dashboard": render_patient_dashboard()
    elif page == "Doctor Management": render_doctor_management()
    elif page == "Appointments": render_appointments_page()
    elif page == "Medical Records": render_medical_records_page()
    elif page == "Billing & Payments": render_billing_page()
    elif page == "Insurance Claims": render_insurance_claims_page()
    elif page == "Secure Messaging": render_secure_messaging_page()
    elif page == "Patient Support": render_patient_support_page()
    elif page == "Emergency & Location": render_emergency_page()

    # Render Chatbot for Patients
    render_floating_chatbot()


# ---- Patient Page Rendering Functions ----
def render_patient_dashboard():
    """Renders the patient dashboard content."""
    profile = st.session_state.patient_profile
    st.subheader(f"👤 Welcome, {profile.get('name', 'Patient')}")
    col1, col2 = st.columns(2)

    with col1: # Profile Card
        st.markdown("### Your Profile")
        with st.container(border=True):
            st.write(f"**Name:** {profile.get('name', 'N/A')}")
            st.write(f"**Age:** {profile.get('age', 'N/A')}")
            st.write(f"**Gender:** {profile.get('gender', 'N/A')}")
            st.write(f"**Username:** {profile.get('username', 'N/A')}")

    with col2: # Favorites Card
        st.markdown("### ❤️ Favorite Doctors")
        with st.container(border=True):
            if st.session_state.favorites:
                # Get details for favorite doctor IDs
                fav_docs = [get_doctor_details(fid) for fid
                            in st.session_state.favorites if get_doctor_details(fid)] # noqa E501
                if fav_docs:
                    for doc in fav_docs:
                        c1, c2 = st.columns([1, 4])
                        c1.image(doc.get('photo_url',''), width=40,
                                 caption=doc.get('name', '').split()[-1])
                        c2.write(f"**{doc.get('name','N/A')}**\n{doc.get('specialty', 'N/A')}") # noqa E501
                        st.divider()
                else: st.info("Could not load favorite doctor details.")
            else: st.info("You haven't added favorite doctors.")

    st.markdown("### 📅 Upcoming Appointments")
    now = datetime.now()
    upcoming_appts = [
        appt for appt in st.session_state.appointments
        if appt.get('datetime') and appt['datetime'] > now
        and appt.get('status', 'Scheduled') != 'Cancelled'
    ]
    if upcoming_appts:
        df_data = []
        for appt in sorted(upcoming_appts, key=lambda x: x['datetime']):
            doc_name = get_doctor_name(appt.get('doctor_id'))
            doc = get_doctor_details(appt.get('doctor_id'))
            specialty = doc.get('specialty', 'N/A') if doc else 'N/A'
            df_data.append({
                "Date & Time": appt['datetime'].strftime('%b %d, %Y %I:%M %p'),
                "Doctor": doc_name, "Specialty": specialty,
                "Status": appt.get("status", "Scheduled")
            })
        st.dataframe(pd.DataFrame(df_data), use_container_width=True, hide_index=True) # noqa E501
    else: st.info("No upcoming appointments scheduled.")


def render_doctor_management():
    """Renders doctor discovery, allows favoriting (updates DB)."""
    st.subheader("🧑‍⚕️ Doctor Discovery")
    st.write("Search doctors, view profiles, manage favorites, contact.")
    search_query = st.text_input("Search Doctors", placeholder="Name/Specialty") # noqa E501

    # Use cached doctors list if available, else fetch
    doctors_list = st.session_state.get('doctors', [])
    if not doctors_list:
        try:
            doctors_list = list(DOCTORS_COLL.find())
            st.session_state.doctors = doctors_list # Cache it
        except Exception as e:
            st.error(f"Failed to load doctors: {e}")
            doctors_list = []

    if search_query:
        results = [
            d for d in doctors_list if
            search_query.lower() in d.get('name','').lower() or
            search_query.lower() in d.get('specialty','').lower()
        ]
        st.info(f"Found {len(results)} doctor(s).")
    else:
        results = doctors_list

    if not results: st.warning("No doctors found."); return

    for doc in results:
        # Use logical ID or fallback to MongoDB _id as string
        doc_logical_id = doc.get('id', str(doc.get('_id')))
        if not doc_logical_id: continue # Skip doc if no usable ID

        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 3, 2])
            with col1: st.image(doc.get('photo_url', ''), width=80)
            with col2:
                st.subheader(doc.get('name', 'N/A'))
                st.write(f"**Specialty:** {doc.get('specialty', 'N/A')}")
                rating = int(doc.get('rating', 0))
                st.write(f"**Rating:** {'⭐' * rating} ({doc.get('rating')})")
                with st.expander("Contact Info & Actions"):
                    st.write(f"**Email:** {doc.get('email', 'N/A')}")
                    st.write(f"**Phone:** {doc.get('phone', 'N/A')}")
                    contact_cols = st.columns(3)
                    # Email
                    with contact_cols[0]:
                        email = doc.get('email')
                        if email: st.link_button("📧 Email", f"mailto:{email}")
                        else: st.button("📧 Email", disabled=True)
                    # WhatsApp
                    with contact_cols[1]:
                        phone = format_phone_number_for_whatsapp(doc.get('phone')) # noqa E501
                        if phone:
                            wa_url = f"https://wa.me/{phone}?text=Hello Dr {doc.get('name','')}" # noqa E501
                            st.link_button("📱 WhatsApp", wa_url)
                        else: st.button("📱 WhatsApp", disabled=True)
                    # Secure Message
                    with contact_cols[2]:
                        if st.button("💬 Message", key=f"msg_{doc_logical_id}"): # noqa E501
                            st.session_state.go_to_messaging = True
                            st.session_state.compose_to_id = doc_logical_id
                            st.rerun()
            with col3: # Actions
                is_favorite = doc_logical_id in st.session_state.favorites
                fav_label = "❤️ Unfav" if is_favorite else "🤍 Fav"
                fav_type = "primary" if is_favorite else "secondary"
                if st.button(fav_label, key=f"fav_{doc_logical_id}", type=fav_type): # noqa E501
                    user_db_id = st.session_state.current_user_db_id
                    success = False
                    if is_favorite:
                        success = remove_favorite_db(user_db_id, doc_logical_id) # noqa E501
                        if success: st.session_state.favorites.discard(doc_logical_id) # noqa E501
                        st.toast(f"Removed {doc.get('name','Doctor')} from favorites." if success else "Error.") # noqa E501
                    else:
                        success = add_favorite_db(user_db_id, doc_logical_id)
                        if success: st.session_state.favorites.add(doc_logical_id) # noqa E501
                        st.toast(f"Added {doc.get('name','Doctor')} to favorites!" if success else "Error.") # noqa E501
                    if success: st.rerun()
                # Book Appointment
                if st.button("📅 Book Appt", key=f"book_{doc_logical_id}"):
                    st.session_state.go_to_appointments = True
                    st.session_state.prefill_doctor_id = doc_logical_id
                    st.rerun()
        st.divider()


def render_appointments_page():
    """Renders appointment scheduling/history (interacts with DB)."""
    st.subheader("📅 Appointment Management")
    tab1, tab2 = st.tabs(["Schedule New Appointment", "View All Appointments"])

    # --- Schedule New Appointment Tab ---
    with tab1:
        st.markdown("#### Book a New Appointment")
        # Use cached doctor list
        doc_options = {
            f"{d.get('name','N/A')} ({d.get('specialty','N/A')})": d.get('id', str(d.get('_id'))) # noqa E501
            for d in st.session_state.doctors
        }
        doc_list = list(doc_options.keys())
        prefill_doc_id = st.session_state.get('prefill_doctor_id')
        prefill_index = 0
        # Find index for pre-filling doctor selection
        if prefill_doc_id and doc_options:
            try: prefill_index = list(doc_options.values()).index(prefill_doc_id) # noqa E501
            except ValueError: pass # ID not found, default to 0
            st.session_state['prefill_doctor_id'] = None # Clear flag

        if not doc_options: st.warning("No doctors loaded/available.")
        else:
            chosen_doc_display = st.selectbox("Choose Doctor", doc_list, index=prefill_index) # noqa E501
            min_date = date.today() + timedelta(days=1)
            selected_date = st.date_input("Select Date", value=min_date, min_value=min_date) # noqa E501
            selected_time = st.time_input("Select Time", value=time(9, 00), step=timedelta(minutes=30)) # noqa E501

            if st.button("Book Appointment", key="book_appt_final"):
                if chosen_doc_display:
                    doctor_logical_id = doc_options[chosen_doc_display]
                    appointment_dt = datetime.combine(selected_date, selected_time) # noqa E501
                    if appointment_dt <= datetime.now(): st.error("Cannot book in past.") # noqa E501
                    else:
                        # Call DB function to add appointment
                        new_appt_doc = add_appointment_db(
                            st.session_state.current_patient_logical_id,
                            doctor_logical_id, appointment_dt
                        )
                        if new_appt_doc:
                            # Refresh local list and UI
                            st.session_state.appointments.insert(0, new_appt_doc) # noqa E501
                            st.session_state.appointments.sort(key=lambda x: x['datetime'], reverse=True) # noqa E501
                            st.info(f"Simulated Notification: Appt with Dr. {doctor_logical_id} booked.") # noqa E501
                            st.toast(f"Appointment with {get_doctor_name(doctor_logical_id)} booked!", icon="✅") # noqa E501
                            st.rerun()
                else: st.error("Please select a doctor.")

    # --- View All Appointments Tab ---
    with tab2:
        st.markdown("#### Your Appointment History")
        patient_appts = st.session_state.appointments # Use loaded data
        if patient_appts:
            df_data = []; now = datetime.now()
            # Data is already sorted descending by datetime from load_user_data # noqa E501
            for appt in patient_appts:
                doc_name = get_doctor_name(appt.get('doctor_id'))
                status = appt.get("status", "Scheduled")
                is_upcoming = (appt.get('datetime') > now and status == "Scheduled") # noqa E501
                doc = get_doctor_details(appt.get('doctor_id'))
                specialty = doc.get('specialty', 'N/A') if doc else 'N/A'
                df_data.append({
                    "DB_ID": appt['_id'], # Keep DB ID for actions
                    "Date & Time": appt.get('datetime').strftime('%Y-%m-%d %I:%M %p'), # noqa E501
                    "Doctor": doc_name, "Specialty": specialty,
                    "Status": status, "Upcoming": is_upcoming
                })
            df_appts = pd.DataFrame(df_data)
            st.dataframe(df_appts.drop(columns=["DB_ID", "Upcoming"]), use_container_width=True, hide_index=True) # noqa E501

            st.markdown("---")
            st.markdown("#### Actions on Upcoming Appointments")
            upcoming_df = df_appts[df_appts["Upcoming"]]
            if not upcoming_df.empty:
                for _, row in upcoming_df.iterrows():
                    st.write(f"**{row['Date & Time']}** with **{row['Doctor']}**") # noqa E501
                    appt_db_id = row['DB_ID'] # Use MongoDB _id
                    cols = st.columns([1, 1, 3])
                    with cols[0]: # Cancel Button
                        if st.button("Cancel", key=f"cancel_{appt_db_id}", type="secondary"): # noqa E501
                            # Call DB update function
                            if update_appt_status_db(appt_db_id, 'Cancelled'):
                                st.info(f"Simulated Notification: Appt {str(appt_db_id)[:8]} cancelled.") # noqa E501
                                st.toast("Appointment cancelled.")
                                # Reload data to refresh UI
                                user_doc = USERS_COLL.find_one({"_id": st.session_state.current_user_db_id}) # noqa E501
                                if user_doc: load_user_data(user_doc)
                                st.rerun()
                            else: st.error("Failed to cancel appointment in DB.") # noqa E501
                    with cols[1]: # Reschedule Hint
                        if st.button("Reschedule", key=f"resched_{appt_db_id}", type="secondary"): # noqa E501
                            st.info("To reschedule: book a new slot and cancel this one.") # noqa E501
            else: st.info("No upcoming appointments for actions.")
        else: st.info("No appointments found.")


def render_medical_records_page():
    """Renders medical record upload/display (interacts with DB)."""
    st.subheader("📚 Medical Records Management")
    st.write("Upload, view, and manage your medical documents.")

    with st.expander("⬆️ Upload New Record", expanded=False):
        record_type = st.selectbox(
            "Record Type",
            ["Prescription", "X-ray/Image", "Lab Report", "Note", "Other"]
        )
        allowed_types = ["pdf", "png", "jpg", "jpeg", "gif", "bmp", "tiff"]
        uploaded_file = st.file_uploader(
            f"Choose file ({', '.join(allowed_types)})", type=allowed_types
        )
        description = st.text_area("Optional Description or Notes")

        if uploaded_file:
            if st.button(f"Upload {record_type}",
                         key=f"upload_btn_{record_type}"):
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > MAX_FILE_UPLOAD_MB:
                    st.error(f"File size exceeds {MAX_FILE_UPLOAD_MB}MB limit.") # noqa E501
                else:
                    temp_path = os.path.join(
                        TEMP_DIR, f"{uuid.uuid4()}_{uploaded_file.name}"
                    )
                    summary = "Error during processing."
                    file_written = False
                    try:
                        file_content = uploaded_file.read()
                        uploaded_file.seek(0)
                        filename = uploaded_file.name
                        file_type = uploaded_file.type or "application/octet-stream" # noqa E501

                        # Save temporarily for path-based LLM function
                        with open(temp_path, "wb") as f:
                            f.write(file_content)
                        file_written = True

                        # Generate summary using imported function
                        prompt = f"Concise summary of this {record_type}: {filename}." # noqa E501
                        # *** Calling the imported function ***
                        summary = generate_image_description(temp_path, prompt)

                        # Add record metadata to DB
                        new_rec_doc = add_record_db(
                            st.session_state.current_patient_logical_id,
                            record_type, filename, file_type,
                            description, summary, len(file_content) / 1024
                        )

                        if new_rec_doc:
                            # Add metadata doc to session for immediate display
                            # Include content only for image preview in this session # noqa E501
                            new_rec_doc["content_data"] = file_content if file_type.startswith("image/") else None # noqa E501
                            st.session_state.medical_records.insert(0, new_rec_doc) # noqa E501
                            st.session_state.medical_records.sort(key=lambda x: x['uploaded_at'], reverse=True) # noqa E501
                            st.success(f"{record_type} uploaded & summarized.") # noqa E501
                            st.info(f"Simulated Notification: Record '{filename}' uploaded.") # noqa E501
                            st.rerun()
                        else:
                            st.error("Failed to save record metadata to DB.")

                    except ImportError:
                        st.error("AI Service unavailable for summary.")
                        summary = "AI Service unavailable."
                        # Optionally add record without summary here if needed
                    except Exception as e:
                        st.error(f"Processing error: {e}")
                        # Use the default error summary set above
                    finally:
                        # Cleanup temp file
                        if file_written and os.path.exists(temp_path):
                            try: os.remove(temp_path)
                            except OSError as e: st.warning(f"Cannot delete temp file {temp_path}: {e}") # noqa E501

    st.markdown("---")
    st.markdown("#### Your Uploaded Records")
    patient_records = st.session_state.medical_records # Use loaded data
    if patient_records:
        df_data = []
        # Already sorted by load_user_data
        for rec in patient_records:
            df_data.append({
                "DB_ID": rec['_id'],
                "Uploaded On": rec['uploaded_at'].strftime('%Y-%m-%d %H:%M'),
                "Type": rec['type'], "Filename": rec['filename'],
                "Description": rec.get('description', 'N/A'),
                "Size (KB)": f"{rec.get('size_kb', 0):.1f}",
                "file_type": rec.get('file_type', ''),
                # Include potential preview content and storage ref
                "content_data": rec.get('content_data'),
                "storage_ref": rec.get("storage_ref"),
                "AI Summary": rec.get('summary', 'N/A'),
            })
        df_records = pd.DataFrame(df_data)
        st.dataframe(
            df_records[['Uploaded On', 'Type', 'Filename', 'Description', 'Size (KB)', 'AI Summary']], # noqa E501
            use_container_width=True, hide_index=True,
            column_config={
                "Description": st.column_config.TextColumn(width="medium"),
                "AI Summary": st.column_config.TextColumn(width="large"),
            }
        )

        st.markdown("---")
        st.markdown("#### Record Actions")
        if not df_records.empty:
            for _, row in df_records.iterrows():
                exp_title = f"{row['Uploaded On']} - {row['Filename']} ({row['Type']})" # noqa E501
                with st.expander(exp_title):
                    st.write(f"**Description:** {row['Description']}")
                    st.write(f"**AI Summary:** {row['AI Summary']}")
                    st.write(f"**File Type:** {row['file_type']}")
                    st.write(f"**Size:** {row['Size (KB)']} KB")
                    st.caption(f"Storage Ref (Simulated): {row['storage_ref']}") # noqa E501

                    # Preview image if content is available in session state
                    if row['file_type'].startswith("image/") and row['content_data']: # noqa E501
                        try:
                            img = Image.open(io.BytesIO(row['content_data']))
                            st.image(img, caption=f"Preview: {row['Filename']}", use_column_width=True) # noqa E501
                        except Exception as e: st.error(f"Img preview error: {e}") # noqa E501

                    # Download button (simulated - gives metadata only)
                    download_content = f"Metadata for: {row['Filename']}\nRef: {row['storage_ref']}" # noqa E501
                    sim_filename = os.path.splitext(row['Filename'])[0] + "_metadata.txt" # noqa E501
                    st.download_button(
                        label="Download (Simulated Metadata)",
                        data=download_content, file_name=sim_filename,
                        mime='text/plain', key=f"download_{row['DB_ID']}"
                    )
        else: st.info("No records found for actions.")
    else: st.info("No medical records uploaded.")


def render_billing_page():
    """Renders billing history (reads from session state, loaded from DB)."""
    st.subheader("💳 Billing History & Payments")
    st.write("View your billing history.") # Removed payment simulation for clarity # noqa E501

    st.markdown("#### Your Billing History")
    patient_bills = st.session_state.billing_history # Use loaded data

    if patient_bills:
        df_data = []; total_due = 0.0
        # Already sorted by load_user_data
        for bill in patient_bills:
            doc_name = get_doctor_name(bill.get('doctor_id'))
            status = bill.get("status", "Due")
            is_due = status == "Due"
            amount = bill.get('amount', 0.0)
            if is_due: total_due += amount

            df_data.append({
                "DB_ID": bill['_id'],
                "Date": bill.get('date').strftime('%Y-%m-%d'),
                "Doctor": doc_name,
                "Amount (₹)": f"{amount:.2f}",
                "Status": status,
                "IsDue": is_due
            })
        df_bills = pd.DataFrame(df_data)
        st.metric("Total Amount Due", f"₹{total_due:.2f}")
        st.dataframe(df_bills[['Date', 'Doctor', 'Amount (₹)', 'Status']], use_container_width=True, hide_index=True) # noqa E501

        # Payment Actions Section (Simplified - no button shown now)
        st.markdown("---")
        st.markdown("#### Payment Information")
        due_bills_df = df_bills[df_bills["IsDue"]]
        if not due_bills_df.empty:
            st.info("Please contact the clinic administration or use the provided external payment portal (if applicable) to settle outstanding bills.") # noqa E501
            # If a real payment link exists, display it here maybe?
            # for _, row in due_bills_df.iterrows():
            #    st.write(f"- Bill from {row['Doctor']} ({row['Amount (₹)']}) - Due") # noqa E501
        else:
            st.success("No outstanding bills found.")
    else:
        st.info("No billing history found.")


def render_insurance_claims_page():
    """Renders insurance claims (reads from session, adds to DB)."""
    st.subheader("📄 Insurance Claims Management")
    st.write("View and record your insurance claims.")

    # Display existing claims from session state
    claims = st.session_state.insurance_claims
    if claims:
        df_data = []
        # Already sorted by load_user_data
        for claim in claims:
            df_data.append({
                "Claim Date": claim.get('claim_date').strftime('%Y-%m-%d'),
                "Provider": claim.get('provider', 'N/A'),
                "Amount (₹)": f"{claim.get('amount', 0.0):.2f}",
                "Status": claim.get('status', 'Submitted'),
                "Details": claim.get('details', 'N/A'),
            })
        st.dataframe(pd.DataFrame(df_data), use_container_width=True, hide_index=True) # noqa E501
    else: st.info("No insurance claims found.")

    # Form to add a new claim record
    with st.expander("Add New Claim Record"):
        with st.form("new_claim_form"):
            claim_date = st.date_input("Claim Submission Date", date.today())
            provider = st.text_input("Insurance Provider Name*")
            amount = st.number_input("Claim Amount (₹)*", min_value=0.01, step=100.0) # noqa E501
            details = st.text_area("Claim Details (e.g., related service)")
            submitted = st.form_submit_button("Add Claim Record")

            if submitted:
                if provider and amount > 0:
                    # Add claim to DB
                    new_claim_doc = add_claim_db(
                        st.session_state.current_patient_logical_id,
                        claim_date, provider, amount, details
                    )
                    if new_claim_doc:
                        # Update local session list and refresh
                        st.session_state.insurance_claims.insert(0, new_claim_doc) # noqa E501
                        st.session_state.insurance_claims.sort(key=lambda x: x['claim_date'], reverse=True) # noqa E501
                        st.success("Insurance claim record added.")
                        st.rerun()
                else: st.warning("Provider name and amount are required.")


def render_secure_messaging_page():
    """Renders secure messaging (reads session, writes/updates DB)."""
    st.subheader("✉️ Secure Messaging")
    st.write("Communicate directly with your doctors.")

    # Compose New Message Section
    st.markdown("#### Compose New Message")
    doc_recipients = {
        f"Dr. {d.get('name','N/A')} ({d.get('specialty','N/A')})": d.get('id', str(d.get('_id'))) # noqa E501
        for d in st.session_state.doctors
    }
    recipient_list = list(doc_recipients.keys())
    prefill_recipient_id = st.session_state.get('compose_to_id')
    prefill_index = 0
    if prefill_recipient_id and doc_recipients:
        try: prefill_index = list(doc_recipients.values()).index(prefill_recipient_id) # noqa E501
        except ValueError: pass
        st.session_state['compose_to_id'] = None # Clear flag

    if not recipient_list: st.warning("No doctors loaded to message.")
    else:
        selected_recipient_display = st.selectbox("To:", recipient_list, index=prefill_index) # noqa E501
        msg_subject = st.text_input("Subject*", key="msg_subj")
        msg_body = st.text_area("Message*", height=150, key="msg_body")

        if st.button("Send Message", key="send_msg_btn"):
            if selected_recipient_display and msg_subject and msg_body:
                recipient_logical_id = doc_recipients[selected_recipient_display] # noqa E501
                # Add message to DB
                new_msg_doc = add_message_db(
                    st.session_state.current_patient_logical_id,
                    recipient_logical_id, msg_subject, msg_body
                )
                if new_msg_doc:
                    # Update local session list and refresh
                    st.session_state.messages.insert(0, new_msg_doc)
                    st.session_state.messages.sort(key=lambda x: x['timestamp'], reverse=True) # noqa E501
                    st.success("Message Sent!")
                    st.info(f"Simulated Notification: Message sent to Dr. {get_doctor_name(recipient_logical_id)}.") # noqa E501
                    st.rerun()
            else: st.warning("Recipient, subject, and message required.")

    # Display Messages
    st.markdown("---")
    st.markdown("#### Your Messages")
    messages = st.session_state.messages # Use loaded data
    if messages:
        # Already sorted by load_user_data
        for msg in messages:
            is_sent = msg.get('from_id') == st.session_state.current_patient_logical_id # noqa E501
            other_party_id = msg.get('to_id') if is_sent else msg.get('from_id') # noqa E501
            other_party_name = "Unknown"
            if other_party_id and other_party_id.startswith('d'):
                other_party_name = get_doctor_name(other_party_id)
            elif other_party_id == "support_team": other_party_name = "Support" # noqa E501

            direction = "To" if is_sent else "From"
            ts = msg.get('timestamp').strftime('%Y-%m-%d %H:%M')
            exp_title = f"{ts} | {direction}: {other_party_name} | Sub: {msg.get('subject','')}" # noqa E501
            is_unread = not msg.get('read', False) and not is_sent
            if is_unread: exp_title += " (Unread)"

            with st.expander(exp_title):
                st.markdown(f"**Subject:** {msg.get('subject','')}")
                st.markdown("**Message:**")
                st.markdown(f"> {msg.get('body','').replace('\n', '\n> ')}")
                # Mark as read when expanded
                if is_unread:
                    if mark_message_read_db(msg['_id']):
                        msg['read'] = True # Update local view immediately
                        # No explicit rerun needed here, state change happens
                    else:
                        print(f"Failed to mark message {msg['_id']} as read in DB.") # noqa E501
    else: st.info("You have no messages.")


def render_patient_support_page():
    """Renders patient support ticket submission/viewing (uses DB)."""
    st.subheader("📢 Support Center")
    st.write("Submit inquiries, feedback, or grievances. Track tickets.")

    with st.expander("✉️ Submit New Ticket", expanded=False):
        subject = st.text_input("Subject*", max_chars=100, key="ticket_subj")
        details = st.text_area("Details*", height=150, key="ticket_details")
        if st.button("Submit Ticket", key="submit_ticket_btn"):
            if subject and details:
                new_tkt_doc = add_ticket_db(
                    st.session_state.current_patient_logical_id, subject, details # noqa E501
                )
                if new_tkt_doc:
                    st.session_state.tickets.insert(0, new_tkt_doc)
                    st.session_state.tickets.sort(key=lambda x: x['submitted_at'], reverse=True) # noqa E501
                    st.success("Support ticket submitted.")
                    st.info(f"Simulated Notification: Ticket '{subject}' submitted.") # noqa E501
                    st.rerun()
            else: st.warning("Subject and details required.")

    st.markdown("---")
    st.markdown("#### Your Support Tickets")
    patient_tickets = st.session_state.tickets # Use loaded data
    if patient_tickets:
        df_data = []
        # Already sorted by load_user_data
        for tk in patient_tickets:
            df_data.append({
                "DB_ID": tk['_id'],
                "Submitted On": tk.get('submitted_at').strftime('%Y-%m-%d %H:%M'), # noqa E501
                "Subject": tk.get('subject', 'N/A'),
                "Status": tk.get('status', 'N/A'),
            })
        df_tickets = pd.DataFrame(df_data)
        st.dataframe(df_tickets[['Submitted On', 'Subject', 'Status']], use_container_width=True, hide_index=True) # noqa E501

        st.markdown("---")
        st.markdown("#### Ticket Details & Comments")
        if not df_tickets.empty:
            for _, row in df_tickets.iterrows():
                ticket_db_id = row['DB_ID']
                # Find full ticket data in session list
                ticket = next((t for t in patient_tickets if t['_id'] == ticket_db_id), None) # noqa E501
                if ticket:
                    exp_title = f"Ticket: {row['Subject']} ({row['Status']}) - {row['Submitted On']}" # noqa E501
                    with st.expander(exp_title):
                        st.markdown("**Details Provided:**")
                        st.markdown(f"> {ticket.get('details','').replace('\n', '\n> ')}") # noqa E501
                        st.markdown("**Comments/Updates:**")
                        comments = ticket.get("comments", [])
                        if comments:
                            sorted_comments = sorted(comments, key=lambda x: x['timestamp']) # noqa E501
                            for comment in sorted_comments:
                                actor = comment.get("actor", "Support")
                                ts = comment['timestamp'].strftime('%Y-%m-%d %H:%M') # noqa E501
                                st.caption(f"{ts} by {actor}:")
                                st.markdown(f"> _{comment['text'].replace('\n', '\n> ')}_") # noqa E501
                        else: st.caption("No comments yet.")

                        # Add Reply Section
                        st.markdown("---")
                        comment_text = st.text_area("Add a reply:", key=f"comment_{ticket_db_id}") # noqa E501
                        if st.button("Add Reply", key=f"reply_{ticket_db_id}"): # noqa E501
                            if comment_text:
                                actor = st.session_state.patient_profile.get("name", "Patient") # noqa E501
                                if add_ticket_comment_db(ticket_db_id, actor, comment_text): # noqa E501
                                    st.success("Reply added.")
                                    st.info(f"Simulated Notification: Reply to ticket {str(ticket_db_id)[:8]}.") # noqa E501
                                    # Reload data or update locally
                                    user_doc = USERS_COLL.find_one({"_id": st.session_state.current_user_db_id}) # noqa E501
                                    if user_doc: load_user_data(user_doc)
                                    st.rerun()
                                else: st.error("Failed to add reply.")
                            else: st.warning("Cannot add an empty reply.")
        else: st.info("No ticket details.")
    else: st.info("You have no support tickets.")


def render_emergency_page():
    """Renders emergency contacts and nearby services (uses LLM)."""
    st.subheader("📍 Emergency Information & Nearby Services")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Emergency Contacts (Ranchi / India)")
        with st.container(border=True):
            st.write("**🚑 Ambulance:** 102 / 108")
            st.write("**🚓 Police:** 100 / 112")
            st.write("**🔥 Fire:** 101")
            st.write("**👩 Women Helpline:** 1091 / 181")
            st.write("**🧒 Child Helpline:** 1098")

    with col2:
        st.markdown("#### Find Nearby Hospitals (AI Powered)")
        with st.container(border=True):
            default_lat, default_lon = 23.3441, 85.3096 # Ranchi
            lat = st.number_input("Latitude (approx)", value=default_lat, format="%.6f") # noqa E501
            lon = st.number_input("Longitude (approx)", value=default_lon, format="%.6f") # noqa E501

            if st.button("Search Nearby Hospitals", key="nearby_search"):
                prompt = f"List hospitals/clinics near lat {lat}, lon {lon} in Ranchi." # noqa E501
                try:
                    # *** Use imported function directly ***
                    response = generate_llm_response(prompt)
                    st.success("Found nearby services (Simulated AI):")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"AI search failed: {e}")
                    st.markdown("Could not fetch nearby services.")


# ---- Admin App Renderer ----
def render_admin_app():
    """Renders the entire Admin Interface."""
    # -- Admin Sidebar --
    st.sidebar.title("Admin Menu")
    st.sidebar.warning("Admin Mode Activated")
    st.sidebar.markdown("---")
    st.sidebar.write("Use tabs below to manage sections.")
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", key="admin_logout"):
        logout(); st.rerun()

    # -- Admin Main Area (Using Tabs) --
    st.subheader("🛠️ Admin Dashboard")
    st.write("Manage system data and view overall metrics.")
    tab_titles = ["📊 Metrics", "🧑‍⚕️ Manage Doctors", "👤 Manage Patients",
                  "🧾 Manage Billing", "🎫 Manage Tickets"]
    tabs = st.tabs(tab_titles)
    # Assign tabs correctly
    tab_metrics, tab_doctors, tab_patients, tab_billing, tab_tickets = tabs

    # Render content for each tab
    with tab_metrics: render_admin_metrics_tab()
    with tab_doctors: render_admin_doctors_tab()
    with tab_patients: render_admin_patients_tab()
    with tab_billing: render_admin_billing_tab()
    with tab_tickets: render_admin_tickets_tab()


# ---- Admin Page Rendering Functions ----
def render_admin_metrics_tab():
    """Renders the content for the Admin Metrics tab (fetches from DB)."""
    st.markdown("#### Key Performance Indicators")

    try:
        # Fetch data directly from DB for aggregate metrics
        all_appointments = list(APPOINTMENTS_COLL.find({}, {"_id": 1, "doctor_id": 1})) # noqa E501
        all_billing = list(BILLING_COLL.find({}, {"_id": 1, "doctor_id": 1, "amount": 1})) # noqa E501
        all_tickets = list(TICKETS_COLL.find({}, {"_id": 1, "status": 1}))
        all_patients_count = USERS_COLL.count_documents({"role": "patient"})
        docs = list(DOCTORS_COLL.find({}, {"_id": 1, "id": 1, "name": 1})) # Fetch needed fields # noqa E501
        st.session_state.doctors = docs # Update admin's cache if needed

        total_revenue = sum(b.get('amount', 0.0) for b in all_billing)
        tickets_open = sum(1 for t in all_tickets if t.get('status') == 'Open')

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total Revenue", f"₹{total_revenue:.2f}")
        mcol2.metric("Total Appointments", len(all_appointments))
        mcol3.metric("Open Support Tickets", tickets_open)
        mcol1.metric("Registered Patients", all_patients_count)
        mcol2.metric("Registered Doctors", len(docs))

        st.divider()
        st.markdown("#### Doctor Performance")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1: # Revenue Chart
            if docs:
                doc_map = {str(d.get('_id')): d.get('name', 'N/A') for d in docs} # noqa E501
                # Use logical ID 'id' as fallback if _id isn't used in billing
                doc_map_logical = {d.get('id'): d.get('name', 'N/A') for d in docs if d.get('id')} # noqa E501
                doc_names = [d.get('name', 'N/A') for d in docs]
                revenue_per_doctor = [
                    sum(b.get('amount', 0.0) for b in all_billing if b.get('doctor_id') == d.get('id', str(d.get('_id')))) # Match by logical ID # noqa E501
                    for d in docs
                ]
                if any(r > 0 for r in revenue_per_doctor):
                    fig1, ax1 = plt.subplots()
                    ax1.bar(doc_names, revenue_per_doctor, color='#1f77b4')
                    ax1.set_title("Revenue/Doctor"); ax1.set_ylabel("₹")
                    ax1.tick_params(axis='x', rotation=45, labelsize=8)
                    plt.tight_layout(); st.pyplot(fig1)
                else: st.info("No revenue data.")
            else: st.info("No doctors found.")

        with chart_col2: # Appointment Chart
            if docs:
                doc_names = [d.get('name', 'N/A') for d in docs]
                appointments_per_doctor = [
                    len([a for a in all_appointments if a.get('doctor_id') == d.get('id', str(d.get('_id')))]) # noqa E501
                    for d in docs
                ]
                if any(a > 0 for a in appointments_per_doctor):
                    fig2, ax2 = plt.subplots()
                    ax2.bar(doc_names, appointments_per_doctor, color='#ff7f0e') # noqa E501
                    ax2.set_title("Appts/Doctor"); ax2.set_ylabel("#")
                    ax2.tick_params(axis='x', rotation=45, labelsize=8)
                    plt.tight_layout(); st.pyplot(fig2)
                else: st.info("No appointment data.")
            else: st.info("No doctors found.")

    except Exception as e:
        st.error(f"Error loading metrics from database: {e}")


def render_admin_doctors_tab():
    """Admin tab for managing doctors in MongoDB."""
    st.markdown("#### Doctor Management")
    st.write("Add, view, or delete doctor profiles.")

    try:
        doctors_list = list(DOCTORS_COLL.find())
        st.session_state.doctors = doctors_list # Update cache
    except Exception as e:
        st.error(f"Failed to load doctors from DB: {e}")
        doctors_list = []

    if doctors_list:
        # Prepare for display, convert ObjectId to str for safety
        display_docs = []
        for doc in doctors_list:
             d = doc.copy()
             d['_id'] = str(d['_id'])
             display_docs.append(d)
        st.dataframe(
            pd.DataFrame(display_docs).drop(columns=['photo_url', '_id'], errors='ignore'), # noqa E501
            use_container_width=True, hide_index=True
        )
    else: st.info("No doctors currently listed.")

    # Delete Doctor
    st.markdown("---"); st.markdown("##### Delete Doctor")
    if doctors_list:
        doc_options = {doc['name']: str(doc['_id']) for doc in doctors_list} # noqa E501
        doc_name_to_delete = st.selectbox(
            "Select Doctor to Delete", options=doc_options.keys(),
            index=None, placeholder="Select...", key="del_doc_sel"
        )
        if doc_name_to_delete:
            st.markdown('<style>.delete-button button {background-color: #dc3545; border-color: #dc3545; color: white;} .delete-button button:hover {background-color: #c82333; border-color: #bd2130;}</style>', unsafe_allow_html=True) # noqa E501
            if st.button("Delete Doctor Permanently", key="del_doc_btn", type="primary", help="This action cannot be undone!"): # noqa E501
                doc_db_id_to_delete = doc_options[doc_name_to_delete]
                try:
                    result = DOCTORS_COLL.delete_one({"_id": ObjectId(doc_db_id_to_delete)}) # noqa E501
                    if result.deleted_count > 0:
                        st.success(f"Doctor '{doc_name_to_delete}' deleted from DB.") # noqa E501
                        st.info("Associated records not deleted.")
                        st.rerun()
                    else: st.warning("Doctor not found in DB or already deleted.") # noqa E501
                except Exception as e: st.error(f"DB error deleting doctor: {e}") # noqa E501
    else: st.info("No doctors to delete.")

    # Add Doctor
    st.markdown("---"); st.markdown("##### Add New Doctor")
    with st.form("new_doctor_form"):
        doc_name = st.text_input("Name*"); doc_spec = st.text_input("Specialty*") # noqa E501
        doc_email = st.text_input("Email"); doc_phone = st.text_input("Phone")
        doc_rating = st.number_input("Rating", 1.0, 5.0, 4.0, 0.1)
        submitted = st.form_submit_button("Add Doctor")
        if submitted:
            if doc_name and doc_spec:
                if doc_email and not is_valid_email(doc_email): st.warning("Invalid email.") # noqa E501
                photo_seed = ''.join(filter(str.isalnum, doc_name)) or "NewDoc"
                doc_photo = f"https://api.dicebear.com/7.x/avataaars/svg?seed={photo_seed}" # noqa E501
                new_doc_data = {
                    # Use a logical 'id' if needed, else rely on _id
                    "id": "d" + str(uuid.uuid4())[:4],
                    "name": doc_name, "specialty": doc_spec,
                    "rating": doc_rating, "photo_url": doc_photo,
                    "email": doc_email, "phone": doc_phone,
                    "created_at": datetime.now()
                }
                try:
                    DOCTORS_COLL.insert_one(new_doc_data)
                    st.success(f"Doctor '{doc_name}' added to DB.")
                    st.rerun()
                except Exception as e: st.error(f"DB error adding doctor: {e}") # noqa E501
            else: st.error("Name and Specialty required.")


def render_admin_patients_tab():
    """Admin tab for managing patients (reads/writes USERS_COLL)."""
    st.markdown("#### Patient Management")
    st.write("Add, view patient credentials.")

    try:
        patient_docs = list(USERS_COLL.find({"role": "patient"}, {"_id": 1, "username": 1, "profile": 1, "patient_id": 1})) # noqa E501
    except Exception as e:
        st.error(f"Failed to load patients from DB: {e}")
        patient_docs = []

    if patient_docs:
        display_list = []
        for pdoc in patient_docs:
            profile = pdoc.get("profile", {})
            display_list.append({
                "DB ID": str(pdoc['_id']),
                "Username": pdoc.get("username"),
                "Logical ID": pdoc.get("patient_id"),
                "Name": profile.get("name"),
                "Age": profile.get("age"),
                "Gender": profile.get("gender"),
            })
        st.dataframe(pd.DataFrame(display_list), use_container_width=True, hide_index=True) # noqa E501
    else: st.info("No patients found in database.")

    # Add Patient
    st.markdown("---"); st.markdown("##### Add New Patient")
    with st.form("new_patient_form"):
        pat_uname = st.text_input("Username (lowercase, unique)*").lower()
        pat_pass = st.text_input("Password*", type="password")
        pat_name = st.text_input("Full Name*")
        pat_age = st.number_input("Age*", 0, 120, 1)
        pat_gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
        pat_logical_id = st.text_input("Logical ID (e.g., p3, optional)", value="p" + str(uuid.uuid4())[:4]) # noqa E501
        submitted = st.form_submit_button("Add Patient")
        if submitted:
            if pat_uname and pat_pass and pat_name and pat_age is not None:
                if not pat_uname.islower() or not pat_uname.isalnum():
                    st.error("Username must be lowercase alphanumeric.")
                else:
                    try:
                        # Check if username exists
                        if USERS_COLL.find_one({"username": pat_uname}):
                            st.error(f"Username '{pat_uname}' already exists.")
                        else:
                            hashed = hash_password(pat_pass)
                            new_user_doc = {
                                "username": pat_uname,
                                "hashed_password": hashed,
                                "role": "patient",
                                "patient_id": pat_logical_id,
                                "profile": {
                                    "name": pat_name, "age": pat_age,
                                    "gender": pat_gender
                                },
                                "created_at": datetime.now()
                            }
                            USERS_COLL.insert_one(new_user_doc)
                            # Initialize sample data structure? Optional here.
                            st.success(f"Patient '{pat_name}' added to DB.")
                            st.rerun()
                    except Exception as e: st.error(f"DB error adding patient: {e}") # noqa E501
            else: st.error("Fill all required fields (*).")


def render_admin_billing_tab():
    """Admin tab for managing billing (reads/writes BILLING_COLL)."""
    st.markdown("#### Billing Management")
    st.write("View all bills and update status.")

    try:
        all_billing = list(BILLING_COLL.find().sort("date", pymongo.DESCENDING)) # noqa E501
    except Exception as e:
        st.error(f"Failed to load billing data: {e}")
        all_billing = []

    if all_billing:
        df_data = []
        for bill in all_billing:
            doc_name = get_doctor_name(bill.get('doctor_id'))
            pat_name = "Unknown"
            # Fetch patient name based on logical ID stored in bill
            if bill.get('patient_id'):
                p_user = USERS_COLL.find_one({"patient_id": bill['patient_id'], "role": "patient"}) # noqa E501
                if p_user: pat_name = p_user.get("profile", {}).get("name", "N/A") # noqa E501
            df_data.append({
                "DB_ID": bill['_id'],
                "Patient ID": bill.get('patient_id'), "Patient Name": pat_name,
                "Date": bill.get('date').strftime('%Y-%m-%d'),
                "Doctor": doc_name,
                "Amount (₹)": f"{bill.get('amount', 0.0):.2f}",
                "Status": bill.get("status", "Due"),
            })
        df_all_bills = pd.DataFrame(df_data)
        st.dataframe(df_all_bills.drop(columns=['DB_ID']), use_container_width=True, hide_index=True) # noqa E501

        # Update Status
        st.markdown("---"); st.markdown("##### Update Bill Status")
        if not df_all_bills.empty:
            bill_dbid_to_update = st.selectbox(
                "Select Bill DB ID to Update",
                options=[str(b['_id']) for b in all_billing], # Use actual DB IDs
                index=None, placeholder="Select...", key="bill_upd_sel"
            )
            new_status = st.selectbox("Set Status", ["Due", "Paid", "Cancelled", "Refunded"], key="bill_status_update") # noqa E501
            if bill_dbid_to_update and st.button("Update Bill Status"):
                try:
                    result = BILLING_COLL.update_one(
                        {"_id": ObjectId(bill_dbid_to_update)},
                        {"$set": {"status": new_status}}
                    )
                    if result.modified_count > 0:
                        st.success(f"Bill {bill_dbid_to_update} status updated.") # noqa E501
                        st.rerun()
                    else: st.warning("Bill not found or status unchanged.")
                except Exception as e: st.error(f"DB error updating bill: {e}") # noqa E501
        else: st.info("No bills to update.")
    else: st.info("No billing records found.")

    # Add Bill Entry
    st.markdown("---"); st.markdown("##### Add New Bill Entry")
    try: # Fetch patients for dropdown
        patient_docs = list(USERS_COLL.find({"role": "patient"}, {"patient_id": 1, "profile.name": 1})) # noqa E501
        pat_dict = {p.get("profile",{}).get("name","N/A"): p.get("patient_id") for p in patient_docs} # noqa E501
    except Exception as e:
        st.error(f"Failed to load patients: {e}")
        pat_dict = {}

    if not pat_dict: st.warning("No patients found to bill.")
    else:
        amt = st.number_input("Amount (₹)*", 0.01, step=50.0, key="admin_bill_amt") # noqa E501
        pat_sel_name = st.selectbox("Patient*", list(pat_dict.keys()), key="admin_bill_pat") # noqa E501
        # Use cached doctors list
        doc_opts = {d.get('name','N/A'): d.get('id', str(d.get('_id'))) for d in st.session_state.doctors} # noqa E501
        if not doc_opts: st.warning("No doctors loaded.")
        else:
            doc_sel_name = st.selectbox("Doctor*", list(doc_opts.keys()), key="admin_bill_doc") # noqa E501
            bill_status = st.selectbox("Initial Status", ["Due", "Paid"])
            if st.button("Add Bill Entry"):
                if amt > 0 and pat_sel_name and doc_sel_name:
                    pat_logical_id = pat_dict[pat_sel_name]
                    doc_logical_id = doc_opts[doc_sel_name]
                    new_bill_doc = {
                        "patient_id": pat_logical_id, "doctor_id": doc_logical_id, # noqa E501
                        "amount": amt, "date": datetime.now(), "status": bill_status # noqa E501
                    }
                    try:
                        BILLING_COLL.insert_one(new_bill_doc)
                        st.success(f"Bill added for {pat_sel_name}.")
                        st.rerun()
                    except Exception as e: st.error(f"DB error adding bill: {e}") # noqa E501
                else: st.warning("Amount, patient, doctor required.")


def render_admin_tickets_tab():
    """Admin tab for managing support tickets (reads/writes TICKETS_COLL)."""
    st.markdown("#### Support Ticket Management")
    st.write("View all tickets, add comments, and change status.")

    try:
        all_tickets = list(TICKETS_COLL.find().sort("submitted_at", pymongo.DESCENDING)) # noqa E501
    except Exception as e:
        st.error(f"Failed to load tickets from DB: {e}")
        all_tickets = []

    if all_tickets:
        df_data = []
        for tk in all_tickets:
            pat_name = "Unknown"
            # Fetch patient name if ID exists
            if tk.get('patient_id'):
                p_user = USERS_COLL.find_one({"patient_id": tk['patient_id'], "role": "patient"}) # noqa E501
                if p_user: pat_name = p_user.get("profile", {}).get("name", "N/A") # noqa E501
            df_data.append({
                "DB_ID": tk['_id'],
                "Patient ID": tk.get('patient_id'), "Patient Name": pat_name,
                "Submitted On": tk.get('submitted_at').strftime('%Y-%m-%d %H:%M'), # noqa E501
                "Subject": tk.get('subject', 'N/A'),
                "Status": tk.get('status', 'N/A'),
            })
        df_all_tickets = pd.DataFrame(df_data)
        st.dataframe(df_all_tickets.drop(columns=['DB_ID']), use_container_width=True, hide_index=True) # noqa E501

        # Manage Specific Ticket
        st.markdown("---"); st.markdown("##### Manage Specific Ticket")
        if not df_all_tickets.empty:
            ticket_dbid_to_manage = st.selectbox(
                "Select Ticket DB ID to Manage",
                options=[str(t['_id']) for t in all_tickets],
                index=None, placeholder="Select...", key="tkt_mng_sel"
            )
            if ticket_dbid_to_manage:
                # Find ticket data again (could optimize by using the list)
                try:
                    ticket_data = TICKETS_COLL.find_one({"_id": ObjectId(ticket_dbid_to_manage)}) # noqa E501
                except Exception as e:
                    st.error(f"Error fetching ticket details: {e}")
                    ticket_data = None

                if ticket_data:
                    patient_name = df_all_tickets[df_all_tickets['DB_ID'] == ObjectId(ticket_dbid_to_manage)]['Patient Name'].iloc[0] # noqa E501
                    st.markdown(f"**Subject:** {ticket_data['subject']} | **Patient:** {patient_name} ({ticket_data.get('patient_id')})") # noqa E501
                    st.markdown("**Original Details:**")
                    st.markdown(f"> {ticket_data.get('details','').replace('\n', '\n> ')}") # noqa E501

                    # Comments
                    st.markdown("**Comments:**")
                    comments = ticket_data.get("comments", [])
                    if comments:
                        for c in sorted(comments, key=lambda x: x['timestamp']): # noqa E501
                            ts = c['timestamp'].strftime('%Y-%m-%d %H:%M')
                            st.caption(f"{ts} by {c.get('actor', 'Support')}:") # noqa E501
                            st.markdown(f"> _{c.get('text','').replace('\n', '\n> ')}_") # noqa E501
                    else: st.caption("No comments.")

                    # Add Admin Comment
                    admin_comment = st.text_area("Add Comment:", key=f"admin_cmt_{ticket_dbid_to_manage}") # noqa E501
                    if st.button("Add Comment", key=f"add_admin_cmt_{ticket_dbid_to_manage}"): # noqa E501
                        if admin_comment:
                            if add_ticket_comment_db(ObjectId(ticket_dbid_to_manage), "Admin Support", admin_comment): # noqa E501
                                st.success("Admin comment added.")
                                st.info(f"Simulated Notification: Update on ticket {ticket_dbid_to_manage[:8]}.") # noqa E501
                                st.rerun()
                        else: st.warning("Empty comment.")

                    # Update Status
                    st.markdown("---")
                    opts = ["Open", "In Progress", "Resolved", "Closed"]
                    try: cur_idx = opts.index(ticket_data.get('status','Open')) # noqa E501
                    except ValueError: cur_idx = 0
                    new_status = st.selectbox("Change Status:", opts, index=cur_idx, key=f"status_{ticket_dbid_to_manage}") # noqa E501
                    if st.button("Update Status", key=f"upd_stat_{ticket_dbid_to_manage}"): # noqa E501
                        if update_ticket_status_db(ObjectId(ticket_dbid_to_manage), new_status): # noqa E501
                            st.success(f"Status updated to '{new_status}'.")
                            st.info(f"Simulated Notification: Status change on ticket {ticket_dbid_to_manage[:8]}.") # noqa E501
                            st.rerun()
                        else: st.error("Failed to update status.")
                else: st.error("Ticket details not found.")
        else: st.info("No tickets to manage.")
    else: st.info("No support tickets found.")


# ---- Main Execution Logic ----
def main():
    """Main function to run the Streamlit application."""
    # Authentication Gate
    if not st.session_state.patient_authenticated and not st.session_state.admin_authenticated: # noqa E501
        # --- Login Screen ---
        st.header("Login")
        login_type = st.radio("Login as:", ["Patient", "Admin"],
                              horizontal=True)

        if login_type == "Patient":
            username = st.text_input("Username", key="login_user").lower()
            password = st.text_input("Password", type="password",
                                     key="login_pass")
            if st.button("Login", key="patient_login_btn"):
                if not username or not password:
                    st.error("Username and password required.")
                else:
                    try:
                        # Query user by username and role
                        user_doc = USERS_COLL.find_one(
                            {"username": username, "role": "patient"}
                        )
                        # Verify password
                        if user_doc and verify_password(password, user_doc.get("hashed_password", "")): # noqa E501
                            st.session_state.patient_authenticated = True
                            st.session_state.admin_authenticated = False
                            load_user_data(user_doc) # Load data into session
                            st.rerun()
                        else:
                            st.error("Invalid patient username or password.")
                    except Exception as e:
                        st.error(f"Login database error: {e}")
                        print(f"Patient Login DB Error: {e}")

        else:  # Admin Login
            admin_username = st.text_input("Admin Username",
                                           key="admin_login_user").lower()
            admin_password = st.text_input("Admin Password", type="password",
                                           key="admin_login_pass")
            if st.button("Login", key="admin_login_btn"):
                if not admin_username or not admin_password:
                    st.error("Admin username and password required.")
                else:
                    try:
                        # Check DB first
                        admin_doc = USERS_COLL.find_one(
                            {"username": admin_username, "role": "admin"}
                        )
                        login_success = False
                        if admin_doc and verify_password(admin_password, admin_doc.get("hashed_password", "")): # noqa E501
                            login_success = True
                        # Fallback to hardcoded only if DB check fails/no admin in DB # noqa E501
                        elif not admin_doc and admin_username == "admin" and ADMIN_CREDENTIALS.get(admin_username) == admin_password: # noqa E501
                             print("Warning: Admin login via hardcoded credentials.") # noqa E501
                             login_success = True

                        if login_success:
                            st.session_state.admin_authenticated = True
                            st.session_state.patient_authenticated = False
                            st.session_state.current_user_db_id = admin_doc["_id"] if admin_doc else "admin_hardcoded" # noqa E501
                            st.session_state.current_patient_logical_id = None
                            st.rerun()
                        else:
                            st.error("Invalid admin username or password.")
                    except Exception as e:
                        st.error(f"Admin login database error: {e}")
                        print(f"Admin Login DB Error: {e}")

    # --- Render App based on Auth Status ---
    elif st.session_state.patient_authenticated:
        render_patient_app()

    elif st.session_state.admin_authenticated:
        render_admin_app()

    else:
        # Should not be reached if logic is correct, but good failsafe
        st.error("Authentication state unclear. Please login again.")
        logout()


if __name__ == "__main__":
    main()