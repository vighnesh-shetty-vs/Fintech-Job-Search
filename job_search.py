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

SEARCH_TERMS = [
    "Data Analyst Intern", 
    "Quantitative Developer Internship", 
    "API Engineer Intern",
    "Machine Learning Stage",
    "Data Science Working Student"
]

# Lowercase countries to prevent JobSpy crashes
LOCATIONS = [
    {"city": "Paris", "country": "france", "needs_sponsorship": False},
    {"city": "Amsterdam", "country": "netherlands", "needs_sponsorship": True},
    {"city": "London", "country": "uk", "needs_sponsorship": True},
    {"city": "Frankfurt", "country": "germany", "needs_sponsorship": True},
    {"city": "Dublin", "country": "ireland", "needs_sponsorship": True},
    {"city": "Luxembourg", "country": "luxembourg", "needs_sponsorship": True}
]

def safe_api_call(prompt, max_retries=3):
    """Executes API call with Exponential Backoff for 100% resilience against drops."""
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
                backoff_time = 20 * (2 ** attempt) # 20s, 40s, 80s
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
                jobs = scrape_jobs(
                    site_name=["linkedin", "indeed"], # Excluded sites that trigger Cloudflare WAF 403s
                    search_term=term,
                    location=loc["city"],
                    results_wanted=30,
                    hours_old=24, 
                    country_relevant=loc["country"]
                )
                
                if not jobs.empty:
                    jobs['search_city'] = loc['city']
                    jobs['needs_sponsorship'] = loc['needs_sponsorship']
                    all_jobs.append(jobs)
                    
            except Exception as e:
                print(f"Error scraping {term} in {loc['city']}: {e}")
            
            # ANTI-BAN MICRO-DELAY: Bypasses datacenter IP rate limiting
            delay = random.uniform(7.0, 14.0)
            print(f"Waiting {round(delay, 2)}s to fly under WAF radar...")
            time.sleep(delay)
                
    if not all_jobs:
        return pd.DataFrame()
        
    df = pd.concat(all_jobs, ignore_index=True)
    df = df.drop_duplicates(subset=['job_url'])
    return df

def apply_hard_filters(df):
    """Filters by applicant count and local keyword matching to save API quota."""
    if 'emails_count' in df.columns:
        df = df[(df['emails_count'] < 10) | (df['emails_count'].isna())]
        
    print("Applying local keyword pre-filter...")
    titles = df['title'].str.lower()
    descs = df['description'].str.lower()
    companies = df['company'].str.lower()
    
    intern_kws = ['intern', 'stage', 'student', 'co-op', 'apprentice', 'alternance']
    has_intern = titles.str.contains('|'.join(intern_kws), na=False) | descs.str.contains('|'.join(intern_kws), na=False)
    
    fin_kws = ['fintech', 'bank', 'financ', 'payment', 'trading', 'quant', 'risk', 'credit', 'wealth', 'asset', 'capital', 'market']
    has_fin = companies.str.contains('|'.join(fin_kws), na=False) | descs.str.contains('|'.join(fin_kws), na=False)
    
    df_filtered = df[has_intern & has_fin].copy()
    print(f"Local filter reduced jobs from {len(df)} to {len(df_filtered)}.")
    return df_filtered

def batch_ai_evaluate(df):
    valid_indices = []
    batch_size = 45 # Mega-batching to drastically reduce total daily requests
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        prompt = "Evaluate these job postings based on the following strict criteria:\n"
        prompt += "1. Domain: Must be strictly Fintech, Banking, or Financial Risk.\n"
        prompt += "2. Role Type: Must be an Internship, Stage, Co-op, or Working Student.\n"
        prompt += "3. Sponsorship: If 'needs_sponsorship' is True, the description MUST explicitly mention visa sponsorship, relocation assistance, or welcoming international applicants.\n\n"
        prompt += "Return ONLY a valid JSON list of the exact 'id' strings that PASS ALL criteria. If none pass, return []. Do not include markdown formatting.\n\n"
        
        jobs_to_evaluate = []
        for idx, row in batch_df.iterrows():
            jobs_to_evaluate.append({
                "id": str(idx),
                "title": row.get('title', ''),
                "company": row.get('company', ''),
                "needs_sponsorship": row.get('needs_sponsorship', False),
                "description": str(row.get('description', ''))[:600] 
            })
            
        prompt += json.dumps(jobs_to_evaluate)
        
        result_text = safe_api_call(prompt)
        
        try:
            passed_ids = json.loads(result_text)
            valid_indices.extend(passed_ids)
            print(f"Batch {i//batch_size + 1} processed successfully.")
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for Batch {i//batch_size + 1}.")
            
        if i + batch_size < len(df):
            time.sleep(15)

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
    msg['Subject'] = f"🚀 Daily Fintech Internship Alerts: {len(df_filtered)} Matches"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    html_content = """
    <html><head><style>
    table { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
    a { color: #1a0dab; text-decoration: none; font-weight: bold; }
    </style></head><body>
    <h2>Targeted Fintech Internships & Stages</h2>
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
    # HUMANIZATION JITTER: Prevents GitHub cron bans
    sleep_time = random.randint(1, 1500) # Random delay between 1 second and 25 minutes
    print(f"Jitter activated. Sleeping for {sleep_time} seconds before execution...")
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
