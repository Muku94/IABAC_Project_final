import streamlit as st
import pandas as pd
import joblib

# Load models and mappings
mappings = joblib.load(r"C:\Users\Muku\OneDrive\Desktop\Data Science\IABAC Exams\IABAC_Projetc_Exam\frequency_mappings.joblib")
scaler = joblib.load(r"C:\Users\Muku\OneDrive\Desktop\Data Science\IABAC Exams\IABAC_Projetc_Exam\scaler_model.joblib")
model = joblib.load(r"C:\Users\Muku\OneDrive\Desktop\Data Science\IABAC Exams\IABAC_Projetc_Exam\rf_model.joblib")

# Define Streamlit layout
st.title('Employee Prediction Model')
st.write('Please enter the following information:')

# Input fields
emp_job_role = st.selectbox('Employee Job Role', [
                'Business Analyst', 'Data Scientist', 'Delivery Manager', 'Developer', 'Finance Manager', 'Healthcare Representative', 'Human Resources', 'Laboratory Technician',
                'Manager', 'Manager R&D', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative', 'Senior Developer',
                'Senior Manager R&D', 'Technical Architect', 'Technical Lead'])
distance_from_home = st.number_input('Distance From Home', min_value=0.0, step=1.0)
emp_education_level = st.number_input('Employee Education Level', min_value=1, max_value=5, step=1)
emp_environment_satisfaction = st.number_input('Employee Environment Satisfaction', min_value=1, max_value=4, step=1)
emp_hourly_rate = st.number_input('Employee Hourly Rate', min_value=0.0, step=1.0)
emp_last_salary_hike_percent = st.number_input('Employee Last Salary Hike Percent', min_value=0, max_value=100, step=1)
emp_work_life_balance = st.number_input('Employee Work Life Balance', min_value=1, max_value=4, step=1)
experience_years_in_current_role = st.number_input('Experience Years in Current Role', min_value=0, step=1)
years_with_curr_manager = st.number_input('Years with Current Manager', min_value=0, step=1)
years_since_last_promotion_log = st.number_input('Years Since Last Promotion (Log)', min_value=0.0, step=0.01)

# Button to make prediction
if st.button('Predict'):
    # Create dataframe from inputs
    new_data = pd.DataFrame({
        'EmpJobRole': [emp_job_role],
        'DistanceFromHome': [distance_from_home],
        'EmpEducationLevel': [emp_education_level],
        'EmpEnvironmentSatisfaction': [emp_environment_satisfaction],
        'EmpHourlyRate': [emp_hourly_rate],
        'EmpLastSalaryHikePercent': [emp_last_salary_hike_percent],
        'EmpWorkLifeBalance': [emp_work_life_balance],
        'ExperienceYearsInCurrentRole': [experience_years_in_current_role],
        'YearsWithCurrManager': [years_with_curr_manager],
        'YearsSinceLastPromotion_log': [years_since_last_promotion_log]
    })

    # Encode categorical columns using the loaded frequency mappings
    for column, mapping in mappings.items():
        if column in new_data.columns:
            new_data[column] = new_data[column].map(mapping).fillna(0)
        else:
            pass

    # Standardize data
    new_data_scaled = scaler.transform(new_data)

    # Make prediction
    prediction = model.predict(new_data_scaled)
    st.write(f'PerformanceRating for this employee is: {prediction[0]}')

