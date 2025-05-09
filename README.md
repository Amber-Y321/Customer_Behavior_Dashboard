[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)](https://streamlit.io)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![GPT-3.5 Turbo](https://img.shields.io/badge/model-GPT--3.5--Turbo-FF8C00)](https://platform.openai.com/docs/models/gpt-3-5-turbo)
![Renvenue Lift](https://img.shields.io/badge/Revenue💰+43.1%25-brightgreen)
[![Session Count Lift](https://img.shields.io/badge/Session%20Lift-+49%25-brightgreen)](https://github.com/your-username/shopify-dashboard#customer-segmentation)

&#x20;

# Shopify Business Analytics Dashboard

> A **Streamlit** application that provides a full-stack analytics for LuxCouture - a Shopify-style E-commerce platform. Built with **Python** and **Streamlit**, it transforms retail and Shopify data into clear and efficient visualization dashboards,  empowering small businesses to optimize marketing strategies and boost popularity and sales.

---
## 🚀 Quickstart

Get up and running in **three steps**:

1. **Clone the repo**
   ```bash
   git clone https://github.com/Amber-Y321/dashboard.git
   cd dashboard
   ```
2. **Install dependencies**
   ```bash
   python3 -m venv .venv       # create virtual environment
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\Activate.ps1 # Windows PowerShell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Launch the app**
   ```bash
   streamlit run dashboard.py
   ```

⭐ *Tip:* Drop your CSVs in `./data/`  before launching.

---
## 🔑 Key Features
> **Interactive Charts**: Hover for detailed deltas and explore daily or weekly performance shifts.

### 🏆 Business Overview
- **Dynamic Sidebar Navigation**: Section selection; Category, subcategory & date Range filter application
   <p align="center">
      <img src="assets/images/Screenshot_sidebar.png" alt="Sidebar Navigation" width="750"/>
   </p>
- **Comprehensive KPIs**  
   - Total Revenue, Order Count, Active vs. Total Customers  
   - Conversion Rate, Return Rate, AOV & LTV
   <p align="center">
      <img src="assets/images/Screenshot_KPI.png" alt="KPI Overview" width="750"/>
   </p>  

- **Sales Analysis**: Revenue breakdown by category with regression-based trend line; subcategory revenue pie chart
   <p align="center">
      <img src="assets/images/Screenshot_Revenue.png" alt="Revenue Analysis" width="750"/>
   </p>  
   
- **Session Trend Analysis**: Online session breakdown by traffic source with linear-regression trend line; online session conversion funnel
   <p align="center">
      <img src="assets/images/Screenshot_Session.png" alt="Session Analysis" width="750"/>
   </p> 

### 🎯 Customer Segmentation
- **Cluster Exploration**: Behavioral clustering and bubble plot based on recency, CLV, high-value ratio, etc.
  <p align="center">
      <img src="assets/images/Screenshot_Customer Segmentation.png" alt="Segment Visualization" width="750"/>
   </p>   
- **AI‑Driven Insights**: GPT-powered assistant suggesting email + social campaigns tailored to seasonal context and segment charateristics via **gpt-3.5-turbo**
![GPT-Driven Interpretation](assets/gif/GIF_GPT.gif)    
---


## 📄 License

Licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.






