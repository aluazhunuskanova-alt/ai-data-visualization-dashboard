from fastapi import FastAPI, UploadFile, File
import pandas as pd
from analysis import analyze_data
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (frontend → backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "API is working"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if file.filename.endswith(".csv"):
            file.file.seek(0)

            # 🔥 TRY AUTO DETECTION FIRST
            try:
                df = pd.read_csv(file.file, sep=None, engine="python", encoding="utf-8")
            except:
                file.file.seek(0)
                df = pd.read_csv(file.file, sep=";", encoding="utf-8")

            # 🔥 CLEAN COLUMN NAMES
            df.columns = df.columns.str.strip()

        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
            df.columns = df.columns.str.strip()

        else:
            return {"error": "Unsupported file format"}

        # 🔍 DEBUG OUTPUT (IMPORTANT)
        print("\n=== COLUMNS ===")
        print(df.columns)

        print("\n=== DATA PREVIEW ===")
        print(df.head())

        print("\n=== DATA TYPES ===")
        print(df.dtypes)

    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}

    if df.empty:
        return {"error": "Dataset is empty"}

    return analyze_data(df)