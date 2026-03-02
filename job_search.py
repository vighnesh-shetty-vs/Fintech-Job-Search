import os
import time
import json
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

# ==========================================
# EXPANDED SEARCH PARAMETERS
# ==========================================
SEARCH_TERMS = [
    "Data Analyst Intern", 
    "Quantitative Developer Internship", 
    "API Engineer Intern",
    "Machine Learning Stage",
    "Data Science Working Student"
]

# FIX: JobSpy requires strictly lowercase country names
LOCATIONS = [
    {"city": "Paris", "country": "france", "needs_sponsorship": False},
    {"city": "Amsterdam", "country": "netherlands", "needs_sponsorship": True},
    {"city": "London", "country": "uk", "needs_sponsorship": True},
    {"city": "Frankfurt", "country": "germany", "needs_sponsorship": True},
    {"city": "Dublin", "country": "ireland", "needs_sponsorship": True},
    {"city": "Luxembourg", "country": "luxembourg", "needs_sponsorship": True}
]

def fetch_jobs():
    all_jobs = []
    for term in SEARCH_TERMS:
        for loc in LOCATIONS:
            print(f"Scraping {term} in {loc['city']}...")
            try:
                # FIX: Removed ZipRecruiter and Glassdoor to prevent Cloudflare 403 WAF blocks on GitHub Actions
                jobs = scrape_jobs(
                    site_name=["linkedin", "indeed"], 
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
                
    if not all_jobs:
        return pd.DataFrame()
        
    # FIX: Suppress future warnings by ensuring clean concatenation
    df = pd.concat(all_jobs, ignore_index=True)
    df = df.drop_duplicates(subset=['job_url'])
    return df

def apply_hard_filters(df):
    if 'emails_count' in df.columns:
        df = df[(df['emails_count'] < 10) | (df['emails_count'].isna())]
    return df

def batch_ai_evaluate(df):
    """
    Evaluates jobs in batches of 15 to completely bypass Gemini API Rate Limits.
    """
    valid_indices = []
    batch_size = 15 
    
    # Iterate over the dataframe in chunks
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        prompt = "Evaluate these job postings based on the following strict criteria:\n"
        prompt += "1. Domain: Must be strictly Fintech, Banking, or Financial Risk.\n"
        prompt += "2. Role Type: Must be an Internship, Stage, Co-op, or Working Student.\n"
        prompt += "3. Sponsorship: If 'needs_sponsorship' is True, the description MUST explicitly mention visa sponsorship, relocation assistance, or welcoming international applicants.\n\n"
        prompt += "Return ONLY a valid JSON list of the exact 'id' strings that PASS ALL criteria. If none pass, return []. Do not include markdown formatting or any other text.\n\n"
        
        jobs_to_evaluate = []
        for idx, row in batch_df.iterrows():
            jobs_to_evaluate.append({
                "id": str(idx),
                "title": row.get('title', ''),
                "company": row.get('company', ''),
                "needs_sponsorship": row.get('needs_sponsorship', False),
                "description": str(row.get('description', ''))[:800] # Truncated to speed up AI processing
            })
            
        prompt += json.dumps(jobs_to_evaluate)
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config={"temperature": 0.0} # Absolute 0 ensures strictly factual JSON output
            )
            
            # Clean up the response safely 
            result_text = response.text.strip().replace('```json', '').replace('```', '')
            passed_ids = json.loads(result_text)
            valid_indices.extend(passed_ids)
            
        except Exception as e:
            print(f"Batch AI Error on chunk {i//batch_size + 1}: {e}")
            
        print(f"Processed batch {i//batch_size + 1}/{(len(df)//batch_size) + 1}. Pausing 12s to respect API RPM limits...")
        time.sleep(12) # Guarantees a maximum of 5 requests per minute, well under the limit.

    # Reconstruct the final dataframe based on the matched indices
    final_indices = []
    for v_id in valid_indices:
        try:
            final_indices.append(int(v_id) if pd.api.types.is_numeric_dtype(df.index) else v_id)
        except:
            pass
            
    return df.loc[df.index.isin(final_indices)]

def send_email(df_filtered):
    if df_filtered.empty:
        print("No jobs matched the strict criteria today. Skipping email.")
        return

    msg = EmailMessage()
    msg['Subject'] = f"🚀 Daily Fintech Internship Alerts: {len(df_filtered)} Matches"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    html_content = """
    <html>
      <head>
        <style>
          table { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #f2f2f2; }
          a { color: #1a0dab; text-decoration: none; font-weight: bold; }
        </style>
      </head>
      <body>
        <h2>Targeted Fintech Internships & Stages</h2>
        <p>Filtered for <10 applicants, posted in the last 24h across major European hubs.</p>
        <table>
          <tr>
            <th>Role</th>
            <th>Company</th>
            <th>Location</th>
            <th>Link</th>
          </tr>
    """
    
    for _, row in df_filtered.iterrows():
        html_content += f"""
          <tr>
            <td>{row.get('title', 'N/A')}</td>
            <td>{row.get('company', 'N/A')}</td>
            <td>{row.get('location', row.get('search_city', 'N/A'))}</td>
            <td><a href="{row.get('job_url', '#')}">Apply Here</a></td>
          </tr>
        """
        
    html_content += """
        </table>
      </body>
    </html>
    """
    
    msg.set_content("Please enable HTML to view this email.")
    msg.add_alternative(html_content, subtype='html')

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("Email sent successfully! Check your inbox.")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    print("Starting AI Job Agent...")
    raw_jobs = fetch_jobs()
    
    if not raw_jobs.empty:
        print(f"Found {len(raw_jobs)} total jobs. Applying hard filters...")
        filtered_jobs = apply_hard_filters(raw_jobs)
        
        if not filtered_jobs.empty:
            print(f"Running Gemini AI batch evaluation on {len(filtered_jobs)} jobs...")
            final_jobs = batch_ai_evaluate(filtered_jobs)
            print(f"{len(final_jobs)} jobs passed all AI checks.")
            send_email(final_jobs)
        else:
            print("No jobs passed the applicant count filter today.")
    else:
        print("No jobs scraped today.")
