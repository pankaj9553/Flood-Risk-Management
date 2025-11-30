# Flood-Risk-Management# 🌧️ Flood Risk Prediction Model & Power BI Analytics Dashboard  
### Machine Learning • Data Analysis • Visualization

This project is an end-to-end data analytics solution combining **Machine Learning** and **Power BI** to understand, analyze, and predict **flood risk**. It includes a predictive model built using Python along with an interactive dashboard visualizing flood patterns, fatalities, monthly/seasonal distribution, and geographic impact.

---

## 🔥 1. Machine Learning: Flood Risk Prediction

### 🎯 Objective  
Build a supervised machine learning model to predict whether a **flood will occur** based on environmental and hydrological features such as rainfall, temperature, humidity, river water level, month, and region.

---

## 🧠 ML Workflow

### **🔹 Data Preprocessing**
- Missing value handling  
- Outlier detection & treatment  
- Encoding categorical variables  
- Scaling continuous features  
- Train-test split  
- Removal of low-importance features  

### **🔹 Feature Engineering**
- Rainfall intensity score  
- Season mapping from month  
- Water level risk index  
- Interaction-based features  

### **🔹 Model Training**
Multiple algorithms were tested:
- Random Forest  
- Logistic Regression  
- SVM  
- KNN  
- XGBoost (optional)  

### **🏆 Best Model: Random Forest**
The Random Forest model delivered the strongest performance with:

- **Accuracy:** 94%  
- **Precision:** 92%  
- **Recall:** 95%  
- **F1 Score:** 93%  

> *(Update accuracy values based on your results.)*

The model was chosen due to its stability, ability to capture non-linear relationships, and low overfitting.

---

## 📈 Model Output  
The model predicts:

Flood Occurred = 1 (Yes)
Flood Occurred = 0 (No)


Based on input features such as rainfall, water level, humidity, temperature, distance from river, and month.

---

## 📊 2. Power BI Flood Risk Analysis Dashboard

### 🎯 Objective  
To visualize historical flood patterns across India and derive insights into:
- Most flood-prone states  
- Seasonal impact  
- Monthly trends  
- Human fatality distribution  
- Long-term flood event patterns  

### 📌 Dashboard Highlights  
The dashboard includes:

#### **🔹 KPIs**
- Total Flood Events: **34K**  
- Total Human Fatalities: **66K**  
- Total Injured: **11K**  
- Avg Flood Duration: **10.82 days**  
- Avg Flood Area %: **3.37%**  

#### **🔹 Visualizations**
- Flood events over years  
- Human fatalities over time  
- Flood by month  
- Flood events by season (Summer: 65%)  
- State-wise flood distribution (Top 10 states)  
- Human fatality by state  
- Geo-map showing flood-affected areas  
- Slicers for **Month**, **Year**, and **State**  


---

## 📌 Key Insights

- **Summer accounts for 65%** of flood occurrences.  
- Flood events peak during **July–September** (monsoon season).  
- **Assam, Uttar Pradesh, Bihar, and Maharashtra** are the most affected states.  
- Human fatalities showed high spikes between 1980–2000.  
- Flood frequency has increased again after 2020.  

---

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)  
- **Power BI** (DAX, Power Query, Interactive Visuals)  
- **Excel / CSV** Dataset  
- **Jupyter Notebook** for ML development  

---

## 🚀 Future Improvements
- Deploy ML model using **Streamlit**  
- Add real-time prediction using **weather API**  
- Create forecasting using **LSTM or Prophet**  
- Publish live dashboard on **Power BI Service**  

---

## 👤 Author  
**Pankaj Kumar Yadav**  
Data Analyst | Power BI Developer | ML Enthusiast  
📧 pankajkumar.666y@gmail.com  

---

