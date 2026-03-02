import os
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
# Targeted for 6-month placements starting May 2026
SEARCH_TERMS = [
    "Data Analyst Intern", 
    "Quantitative Developer Internship", 
    "API Engineer Intern",
    "Machine Learning Stage",
    "Data Science Working Student"
]

# Expanded Financial Hubs
LOCATIONS = [
    {"city": "Paris", "country": "France", "needs_sponsorship": False},
    {"city": "Amsterdam", "country": "Netherlands", "needs_sponsorship": True},
    {"city": "London", "country": "UK", "needs_sponsorship": True},
    {"city": "Frankfurt", "country": "Germany", "needs_sponsorship": True},
    {"city": "Dublin", "country": "Ireland", "needs_sponsorship": True},
    {"city": "Luxembourg", "country": "Luxembourg", "needs_sponsorship": True}
]

def fetch_jobs():
    all_jobs = []
    for term in SEARCH_TERMS:
        for loc in LOCATIONS:
            print(f"Scraping {term} in {loc['city']}...")
            try:
                jobs = scrape_jobs(
                    site_name=["linkedin", "indeed", "glassdoor", "zip_recruiter"],
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
        
    df = pd.concat(all_jobs, ignore_index=True)
    df = df.drop_duplicates(subset=['job_url'])
    return df

def apply_hard_filters(df):
    if 'emails_count' in df.columns:
        df = df[(df['emails_count'] < 10) | (df['emails_count'].isna())]
    return df

def ai_evaluate_job(title, company, description, needs_sponsorship):
    desc_snippet = str(description)[:1500] 
    
    prompt = f"""
    Evaluate the following job posting:
    Title: {title}
    Company: {company}
    Description: {desc_snippet}
    
    Task 1: Is this company or role strictly in the Fintech, Banking, or Financial Risk domain?
    Task 2: Is this explicitly an Internship, Stage, Co-op, or Working Student role?
    Task 3: If 'needs_sponsorship' is True ({needs_sponsorship}), does the description explicitly mention offering visa sponsorship, relocation assistance, or welcoming international applicants?
    
    Reply ONLY in this exact format:
    Fintech: [Yes/No]
    Internship: [Yes/No]
    Sponsorship: [Yes/No/Not Required]
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
            config={"temperature": 0.1} 
        )
        
        result = response.text.strip()
        is_fintech = "Fintech: Yes" in result
        is_internship = "Internship: Yes" in result
        
        if needs_sponsorship:
            has_sponsorship = "Sponsorship: Yes" in result
            return is_fintech and is_internship and has_sponsorship
            
        return is_fintech and is_internship
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return False

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
        
        print(f"Running Gemini AI evaluation on {len(filtered_jobs)} jobs...")
        valid_indices = []
        for index, row in filtered_jobs.iterrows():
            if ai_evaluate_job(row['title'], row['company'], row['description'], row['needs_sponsorship']):
                valid_indices.append(index)
                
        final_jobs = filtered_jobs.loc[valid_indices]
        print(f"{len(final_jobs)} jobs passed all AI checks.")
        
        send_email(final_jobs)
    else:
        print("No jobs scraped today.")