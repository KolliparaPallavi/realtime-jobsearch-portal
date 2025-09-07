# Job Search Web Application  

A simple **job search platform** built with Flask that helps **job seekers** find jobs and allows **employers** to create new job listings.  
It also uses a **machine learning model** to recommend job categories for new listings, improving search relevance and reducing manual errors.  


## ✨ Features  

### For Job Seekers  
- Search jobs using keywords (e.g., "work", "worked", "works" → same results).  
- View number of matching job ads.  
- Preview job listings and open full job descriptions.  

### For Employers  
- Create new job adverts with title, company, salary, and description.  
- Get **AI-powered job category recommendations**.  
- Override categories if needed.  
- New listings are instantly searchable.  



## 🛠️ Technologies Used  

- **Flask** – Web framework  
- **Python (NLTK)** – Text preprocessing (stemming, tokenization)  
- **Gensim FastText** – Word embeddings for job classification  
- **Scikit-learn** (Pickle model) – Machine learning for category prediction  
- **HTML & Jinja2** – Frontend templates for job browsing  
- **Pandas & NumPy** – Data handling and numerical operations  
- **OS & File I/O** – Store and retrieve job listings  



## 🚀 How It Works  

1. **Job seekers** enter keywords → System finds relevant job ads.  
2. **Employers** create listings → System suggests categories using ML/NLP.  
3. Jobs are stored in structured folders and searchable right away.  



