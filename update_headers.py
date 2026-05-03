import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

CREDENTIALS_FILE = "credentials.json"
SHEET_ID_FILE = "spreadsheet_id.txt"

creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
svc = build("sheets", "v4", credentials=creds)

with open(SHEET_ID_FILE) as f:
    sid = f.read().strip()

HEADERS = [
    "Annotator Type", "Participant ID", "Scene ID",
    "Environment", "Lighting",
    "Traffic Density", "Traffic Flow",
    "Total Vehicles", "Total Pedestrians", "Total Cyclists",
    "Total Traffic Lights",
    "Scene Narrative", "Spatial Description", "Hazards and Events"
]

svc.spreadsheets().values().clear(spreadsheetId=sid, range="Sheet1!1:1").execute()
svc.spreadsheets().values().update(
    spreadsheetId=sid,
    range="Sheet1!A1",
    valueInputOption="RAW",
    body={"values": [HEADERS]}
).execute()

print("Headers successfully updated in Sheet1!")
