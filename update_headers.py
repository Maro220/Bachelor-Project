
from google.oauth2 import service_account
from googleapiclient.discovery import build

from scene_annotator import SHEET_HEADERS

CREDENTIALS_FILE = "credentials.json"
SHEET_ID_FILE    = "spreadsheet_id.txt"

creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
svc   = build("sheets", "v4", credentials=creds)

with open(SHEET_ID_FILE) as f:
    sid = f.read().strip()

last_col = chr(ord("A") + len(SHEET_HEADERS) - 1)
svc.spreadsheets().values().clear(spreadsheetId=sid, range="Sheet1!1:1").execute()
svc.spreadsheets().values().update(
    spreadsheetId=sid,
    range="Sheet1!A1",
    valueInputOption="RAW",
    body={"values": [SHEET_HEADERS]}
).execute()

print(f"✓ Wrote {len(SHEET_HEADERS)} headers to Sheet1!A1:{last_col}1")
