# Civil Engineering Study Bot (Streamlit + Groq)

## Sabse fast tareeqa (laptop par, 5 minute)
1. Apne PDFs `docs/` folder me copy karo.
2. `.streamlit/secrets.toml.example` ko copy karke `.streamlit/secrets.toml` banao aur apni Groq key daalo.
3. Terminal:
   pip install -r requirements.txt
   streamlit run app.py
4. Browser me khul jayega. Bas.

## Streamlit Cloud par deploy (phone se bhi use karne ke liye)
1. GitHub repo me poora folder push karo (docs/ ke PDFs sameet). secrets.toml push MAT karna.
2. share.streamlit.io -> New app -> repo select -> main file: app.py
3. Advanced settings -> Secrets me likho:  GROQ_API_KEY = "gsk_..."
4. Deploy.

## Tips
- Sidebar me "Answer style": exam revision, numerical solve, ya MCQ practice.
- Rate limit aaye to app.py me MODEL = "llama-3.1-8b-instant" kar do.
- Scanned (photo wale) PDF me text nahi hota, sidebar warning dega.
