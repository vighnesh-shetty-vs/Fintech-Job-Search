import os
import time
import json
import random
import smtplib
import pandas as pd
from email.message import EmailMessage
from jobspy import scrape_jobs
from google import genai
from dotenv import load_dotenv

load_dotenv() 

# ==========================================
# CONFIGURATION & SECRETS
# ==========================================
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") 
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

# Broadened categories leveraging both BI/Data and Backend/API experience
SEARCH_TERMS = [
    # Data & Analytics
    "Data Analyst Intern", 
    "Data Science Working Student",
    "Business Intelligence Intern",
    "BI Stage",
    # Engineering & Backend
    "Data Engineer Intern",
    "API Engineer Intern",
    "Backend Developer Intern",
    # Finance/Quant
    "Quantitative Developer Internship", 
    "Risk Analyst Intern"
]

# Expanded EU/Asia hubs. US/Canada implicitly ignored by omission, 
# and explicitly filtered out in the hard filters below.
LOCATIONS = [
    # France (No sponsorship needed)
    {"city": "Paris", "country": "france", "needs_sponsorship": False},
    {"city": "Lyon", "country": "france", "needs_sponsorship": False},
    # EU Hubs
    {"city": "Amsterdam", "country": "netherlands", "needs_sponsorship": True},
    {"city": "London", "country": "uk", "needs_sponsorship": True},
    {"city": "Frankfurt", "country": "germany", "needs_sponsorship": True},
    {"city": "Berlin", "country": "germany", "needs_sponsorship": True},
    {"city": "Dublin", "country": "ireland", "needs_sponsorship": True},
    {"city": "Luxembourg", "country": "luxembourg", "needs_sponsorship": True},
    {"city": "Zurich", "country": "switzerland", "needs_sponsorship": True},
    {"city": "Milan", "country": "italy", "needs_sponsorship": True},
    # Asia Hubs
    {"city": "Dubai", "country": "uae", "needs_sponsorship": True},
    {"city": "Singapore", "country": "singapore", "needs_sponsorship": True}
]

def safe_api_call(prompt, max_retries=3):
    """Executes API call with Exponential Backoff to respect 15 RPM limits."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config={"temperature": 0.0}
            )
            return response.text.strip().replace('```json', '').replace('```', '')
            
        except Exception as e:
            print(f"API Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                backoff_time = 20 * (2 ** attempt) 
                print(f"Backing off for {backoff_time} seconds...")
                time.sleep(backoff_time)
            else:
                print("Max retries reached. Failing gracefully.")
                return "[]"

def fetch_jobs():
    all_jobs = []
    for term in SEARCH_TERMS:
        for loc in LOCATIONS:
            print(f"Scraping {term} in {loc['city']}...")
            try:
                # Added 'google' Google Jobs is highly resilient to datacenter IPs.
                jobs = scrape_jobs(
                    site_name=["linkedin", "indeed", "google"], 
                    search_term=term,
                    location=loc["city"],
                    results_wanted=20, 
                    hours_old=24, 
                    country_relevant=loc["country"]
                )
                
                if not jobs.empty:
                    jobs['search_city'] = loc['city']
                    jobs['needs_sponsorship'] = loc['needs_sponsorship']
                    all_jobs.append(jobs)
                    
            except Exception as e:
                print(f"Error scraping {term} in {loc['city']}: {e}")
            
            # ANTI-BAN MICRO-DELAY
            delay = random.uniform(8.0, 16.0)
            time.sleep(delay)
                
    # Filter out any empty dataframes before concatenating
    valid_jobs = [j for j in all_jobs if not j.empty]
    
    if not valid_jobs:
        return pd.DataFrame()
        
    df = pd.concat(valid_jobs, ignore_index=True)
    df = df.drop_duplicates(subset=['job_url'])
    return df

def apply_hard_filters(df):
    """Aggressive local filtering to save API quota and enforce constraints."""
    if 'emails_count' in df.columns:
        df = df[(df['emails_count'] < 10) | (df['emails_count'].isna())]
        
    titles = df['title'].astype(str).str.lower()
    descs = df['description'].astype(str).str.lower()
    locations = df['location'].astype(str).str.lower()
    companies = df['company'].astype(str).str.lower()
    
    # 1. Strict Exclusions (No USA/Canada)
    na_kws = ['usa', 'united states', 'canada', 'ny ', 'ca ', 'tx ']
    is_not_na = ~locations.str.contains('|'.join(na_kws), na=False)
    
    # 2. Intern/Stage Requirement
    intern_kws = ['intern', 'stage', 'student', 'co-op', 'apprentice', 'alternance', 'graduate']
    has_intern = titles.str.contains('|'.join(intern_kws), na=False) | descs.str.contains('|'.join(intern_kws), na=False)
    
    # 3. Local Sponsorship Pre-Filter (Saves API calls)
    sponsorship_kws = ['visa', 'sponsor', 'relocat', 'international', 'global mobility', 'permit']
    has_sponsorship_mention = descs.str.contains('|'.join(sponsorship_kws), na=False)
    
    # Keep if: It's France OR (It needs sponsorship AND mentions sponsorship keywords)
    valid_visa = (~df['needs_sponsorship']) | (df['needs_sponsorship'] & has_sponsorship_mention)
    
    # 4. Domain Focus
    fin_kws = ['fintech', 'bank', 'financ', 'payment', 'trading', 'quant', 'risk', 'credit', 'wealth', 'data', 'tech']
    has_fin = companies.str.contains('|'.join(fin_kws), na=False) | descs.str.contains('|'.join(fin_kws), na=False)
    
    df_filtered = df[is_not_na & has_intern & valid_visa & has_fin].copy()
    print(f"Local filter reduced jobs from {len(df)} to {len(df_filtered)}.")
    return df_filtered

def batch_ai_evaluate(df):
    valid_indices = []
    batch_size = 40 
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        prompt = "Evaluate these job postings based on the following strict criteria:\n"
        prompt += "1. Domain: Must be relevant to Analytics, Data Engineering, API Development, or Finance/Fintech.\n"
        prompt += "2. Role Type: Must be an Internship, Stage, Co-op, or Working Student position.\n"
        prompt += "3. Sponsorship: If 'needs_sponsorship' is True, the description MUST explicitly mention visa sponsorship, relocation assistance, or welcoming international applicants.\n\n"
        prompt += "Return ONLY a valid JSON list of the exact 'id' strings that PASS ALL criteria. If none pass, return []. Do not include markdown formatting.\n\n"
        
        jobs_to_evaluate = []
        for idx, row in batch_df.iterrows():
            jobs_to_evaluate.append({
                "id": str(idx),
                "title": row.get('title', ''),
                "company": row.get('company', ''),
                "needs_sponsorship": row.get('needs_sponsorship', False),
                "description": str(row.get('description', ''))[:700] # Slight bump for better context
            })
            
        prompt += json.dumps(jobs_to_evaluate)
        
        result_text = safe_api_call(prompt)
        
        try:
            passed_ids = json.loads(result_text)
            valid_indices.extend(passed_ids)
            print(f"Batch {i//batch_size + 1} processed successfully.")
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for Batch {i//batch_size + 1}. Output was: {result_text[:50]}")
            
        if i + batch_size < len(df):
            time.sleep(16) # Ensures we stay under 15 RPM

    final_indices = []
    for v_id in valid_indices:
        try:
            final_indices.append(int(v_id) if pd.api.types.is_numeric_dtype(df.index) else v_id)
        except:
            pass
            
    return df.loc[df.index.isin(final_indices)]

def send_email(df_filtered):
    if df_filtered.empty:
        print("No jobs matched today. Skipping email.")
        return

    msg = EmailMessage()
    msg['Subject'] = f"🚀 Targeted Intern Alerts: {len(df_filtered)} Matches (Data/Fintech)"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    html_content = """
    <html><head><style>
    table { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
    a { color: #1a0dab; text-decoration: none; font-weight: bold; }
    </style></head><body>
    <h2>Daily Target Matches</h2>
    <table><tr><th>Role</th><th>Company</th><th>Location</th><th>Link</th></tr>
    """
    for _, row in df_filtered.iterrows():
        html_content += f"<tr><td>{row.get('title', 'N/A')}</td><td>{row.get('company', 'N/A')}</td><td>{row.get('location', row.get('search_city', 'N/A'))}</td><td><a href='{row.get('job_url', '#')}'>Apply Here</a></td></tr>"
    html_content += "</table></body></html>"
    
    msg.set_content("Please enable HTML to view this email.")
    msg.add_alternative(html_content, subtype='html')

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    # GitHub Action max runtime is 6 hours. Jitter capped at 15 mins to be safe.
    sleep_time = random.randint(1, 900) 
    print(f"Jitter activated. Sleeping for {sleep_time} seconds to bypass static bot detection...")
    time.sleep(sleep_time)
    
    print("Starting AI Job Agent...")
    raw_jobs = fetch_jobs()
    
    if not raw_jobs.empty:
        filtered_jobs = apply_hard_filters(raw_jobs)
        if not filtered_jobs.empty:
            final_jobs = batch_ai_evaluate(filtered_jobs)
            send_email(final_jobs)
        else:
            print("No jobs passed local filters.")
    else:
        print("No jobs scraped today.")
