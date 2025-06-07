import streamlit as st
st.set_page_config(page_title="Early Thyroid Detector", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pickle


# Load the trained model only (no scaler)
@st.cache_resource
def load_model():
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    return model

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("thyroidDF.csv")
        #st.write("✅ Raw data loaded successfully:", df.shape)

        df.dropna(subset=['target'], inplace=True)
        df['target'] = df['target'].astype(str).str.strip().str.title()

        hyperthyroid_conditions = ['A', 'B', 'C', 'D', 'O', 'P', 'Q', 'S', 'T']
        hypothyroid_conditions = ['E', 'F', 'G', 'H']
        normal_conditions = ['-']

        def categorize_target(value):
            diagnoses = str(value).split('|')
            for diagnosis in diagnoses:
                if diagnosis in hyperthyroid_conditions:
                    return 'Hyperthyroid'
            for diagnosis in diagnoses:
                if diagnosis in hypothyroid_conditions:
                    return 'Hypothyroid'
            for diagnosis in diagnoses:
                if diagnosis in normal_conditions:
                    return 'Negative'
            return 'Unknown'

        df['target'] = df['target'].apply(categorize_target)
        df = df[df['target'].isin(['Hyperthyroid', 'Hypothyroid', 'Negative'])]
        #df = df.fillna(df.median(numeric_only=True))
        if 'sex' in df.columns:
            df['sex'] = df['sex'].fillna('Unknown')
            df['sex'] = df['sex'].astype(str).str.upper().str.strip()
            df['sex'] = df['sex'].map({'F': 0, 'M': 1, 'UNKNOWN': np.nan})  # optional: leave unknowns as NaN
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() == 2:
                df[col] = df[col].map({'f': 0, 't': 1})
       
        return df

    except Exception as e:
        st.error(f"❌ Failed to load or preprocess dataset: {e}")
        return pd.DataFrame()

model = load_model()

df = None
try:
    df = load_data()
    if df is None or df.empty:
        st.error("The dataset is empty after preprocessing. Please check your source file.")
except FileNotFoundError:
    st.error("Dataset not found. Please check the file path: '/content/drive/MyDrive/kaggle/thyroidDF.csv'")
    df = pd.DataFrame()

# Page navigation
page = st.sidebar.selectbox("Choose a page", ["Prediction", "EDA", "About Thyroid"])

if page == "Prediction":
    st.markdown("""
    <div style='text-align:center;'>
        <h1 style='color:#4CAF50;'>🧠 Early Detection System for Hyperthyroidism </h1>
        <p style='font-size:18px;'>Provide patient information below to predict thyroid condition.</p>
    </div>
    <hr style='border: 1px solid #ddd;'>
""", unsafe_allow_html=True)

    with st.form("prediction_form"):
        
        st.markdown("### 🧾 Demographics")
        col1, col2 = st.columns(2)
        st.markdown("""
            <style>
                .stSelectbox, .stNumberInput, .stButton button {
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 16px;
                }
                .stButton button {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                }
                .stButton button:hover {
                    background-color: #45a049;
                }
            </style>
        """, unsafe_allow_html=True)

        

        with col1:
            age = st.number_input("Age", min_value=0, max_value=100, value=30)
        with col2:
            sex = st.selectbox("Sex", options=["Male", "Female"])
        with col1:
            referral_input = st.selectbox("Referral Source", options=['STMW', 'SVHC', 'SVHD', 'SVI', 'WEST', 'other'])

        st.markdown("### 💊 Medication & Treatment")
        col1, col2 = st.columns(2)
        st.markdown("""
            <style>
                .stSelectbox, .stNumberInput, .stButton button {
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 16px;
                }
                .stButton button {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                }
                .stButton button:hover {
                    background-color: #45a049;
                }
            </style>
        """, unsafe_allow_html=True)

        with col1:
            on_thyroxine = st.selectbox("On Thyroxine", options=["False", "True"], key="on_thyroxine")
        with col2:
            query_on_thyroxine = st.selectbox("Query on Thyroxine", options=["False", "True"], key="query_on_thyroxine")
        with col1:
            on_antithyroid_meds = st.selectbox("On Antithyroid Meds", options=["False", "True"], key="on_antithyroid_meds")
        with col2:
            I131_treatment = st.selectbox("I131 Treatment", options=["False", "True"], key="i131_treatment")
        with col1:
            thyroid_surgery = st.selectbox("Thyroid Surgery", options=["False", "True"], key="thyroid_surgery")
        with col2:
            lithium = st.selectbox("Lithium", options=["False", "True"], key="lithium")

        st.markdown("### 🩺 Clinical Indicators")
        col1, col2 = st.columns(2)
        st.markdown("""
            <style>
                .stSelectbox, .stNumberInput, .stButton button {
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 16px;
                }
                .stButton button {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                }
                .stButton button:hover {
                    background-color: #45a049;
                }
            </style>
        """, unsafe_allow_html=True)
        with col1:
            sick = st.selectbox("Sick", options=["False", "True"], key="sick")
        with col2:
            pregnant = st.selectbox("Pregnant", options=["False", "True"], key="pregnant")
        with col1:
            query_hypothyroid = st.selectbox("Query Hypothyroid", options=["False", "True"], key="query_hypothyroid")
        with col2:
            query_hyperthyroid = st.selectbox("Query Hyperthyroid", options=["False", "True"], key="query_hyperthyroid")
        with col1:
            goitre = st.selectbox("Goitre", options=["False", "True"], key="goitre")
        with col2:
            tumor = st.selectbox("Tumor", options=["False", "True"], key="tumor")
        with col1:
            hypopituitary = st.selectbox("Hypopituitary", options=["False", "True"], key="hypopituitary")
        with col2:
            psych = st.selectbox("Psych", options=["False", "True"], key="psych")

        st.markdown("### 🧪 Lab Test Results")
        col1, col2 = st.columns(2)
        st.markdown("""
            <style>
                .stSelectbox, .stNumberInput, .stButton button {
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 16px;
                }
                .stButton button {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                }
                .stButton button:hover {
                    background-color: #45a049;
                }
            </style>
        """, unsafe_allow_html=True)
        with col1:
            TSH = st.number_input("TSH", value=2.5)
        with col2:
            T3 = st.number_input("T3", value=1.5)
        with col1:
            TT4 = st.number_input("TT4", value=100.0)
        with col2:
            T4U = st.number_input("T4U", value=1.0)
        with col1:
            FTI = st.number_input("FTI", value=100.0)

        submit = st.form_submit_button("Predict")

  
    if submit:
        with open("model_columns.pkl", "rb") as f:
            model_columns = pickle.load(f)
        def encode_bool(val):
            return 1 if val == "True" else 0
        referral_options = {
    'STMW': 0,
    'SVHC': 1,
    'SVHD': 2,
    'SVI': 3,
    'WEST': 4,
    'other': 5}
        class_mapping = {0: 'Hyperthyroid', 1: 'Hypothyroid', 2: 'Negative'}

        input_dict = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "on_thyroxine": encode_bool(on_thyroxine),
        "query_on_thyroxine": encode_bool(query_on_thyroxine),
        "on_antithyroid_meds": encode_bool(on_antithyroid_meds),
        "sick": encode_bool(sick),
        "pregnant": encode_bool(pregnant),
        "thyroid_surgery": encode_bool(thyroid_surgery),
        "I131_treatment": encode_bool(I131_treatment),
        "query_hypothyroid": encode_bool(query_hypothyroid),
        "query_hyperthyroid": encode_bool(query_hyperthyroid),
        "lithium": encode_bool(lithium),
        "goitre": encode_bool(goitre),
        "tumor": encode_bool(tumor),
        "hypopituitary": encode_bool(hypopituitary),
        "psych": encode_bool(psych),
        "TSH": TSH,
        "T3": T3,
        "TT4": TT4,
        "T4U": T4U,
        "FTI": FTI,
        "referral_source": referral_options[referral_input]
    }
        input_df = pd.DataFrame([input_dict])  # constructed from user input
        input_df = input_df[model_columns]
        prediction = model.predict(input_df)[0]
        predicted_label = class_mapping.get(prediction, "Unknown")


        st.markdown(f"""
    <div style='padding: 20px; background-color: #e8f5e9; border-radius: 10px; text-align: center;'>
        <h2 style='color: #2e7d32;'>✅ Predicted Thyroid Condition: <b>{predicted_label}</b></h2>
    </div>""", unsafe_allow_html=True)
        st.write("Input DataFrame for Prediction:",input_df)


        

elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Class Distribution",
        "Age Distribution",
        "TSH Levels",
        "Sex Distribution",
        "TT4 Levels",
        "Correlation Heatmap"
    ])

    if df is None or df.empty:
        for tab in [tab1, tab2, tab3, tab4, tab5, tab6]:
            with tab:
                st.warning("EDA cannot be shown because the dataset is not loaded or is empty.")
    else:
        import plotly.express as px

        with tab1:
            st.subheader("Target Class Distribution")
            target_counts = df['target'].value_counts()
            target_df = target_counts.reset_index()
            target_df.columns = ['Diagnosis', 'Count']
            fig1 = px.bar(target_df, x='Diagnosis', y='Count',
                          labels={'Diagnosis': 'Diagnosis', 'Count': 'Count'},
                          title='Distribution of Thyroid Conditions',
                          width=800, height=800)
            st.plotly_chart(fig1)
            st.markdown(
    """
    <div style='font-size:16px; line-height:1.6; padding-top:10px; color:#333;'>
        <strong>This bar chart shows the number of patients in each thyroid condition class:</strong> 
        Hyperthyroid, Hypothyroid, and Negative (normal).<br>
        The majority of patients are classified as having no thyroid disorder (<em>'Negative'</em>), with over 6,000 records.<br>
        In contrast, <em>'Hypothyroid'</em> and <em>'Hyperthyroid'</em> conditions are significantly less common, indicating class imbalance.
    </div>
    """,
    unsafe_allow_html=True
)
        with tab2:
            st.subheader("Age Distribution by Diagnosis")
            filtered_df = df[df['age'] < 100]
            fig2 = px.box(filtered_df, x="target", y="age",
                          title="Age Distribution per Diagnosis", points="all",
                          width=800, height=800)
            st.plotly_chart(fig2)
            st.markdown(
    """
    <div style='font-size:16px; line-height:1.6; padding-top:10px; color:#333;'>
        <strong>This boxplot illustrates the age distribution of patients across different thyroid diagnoses.</strong><br>
        Patients diagnosed as 'Negative' (no thyroid disorder) are spread across a broad age range, with many younger individuals included.<br>
        In contrast, both 'Hyperthyroid' and 'Hypothyroid' groups show more concentration in middle to older age brackets, indicating possible age-related risk factors.<br>
        This boxplot shows how patient age varies across diagnosis categories. Younger and older populations may show different prevalence rates.
    </div>
    """,
    unsafe_allow_html=True
)
        with tab3:
            st.subheader("TSH Levels by Diagnosis")
            filtered_df = df[df['TSH'] < 100]
            fig3 = px.violin(filtered_df, x="target", y="TSH", box=True, points="all",
                             title="TSH Level Distribution", width=800, height=800)
            st.plotly_chart(fig3)
            st.markdown(
    """
    <div style='font-size:16px; line-height:1.6; padding-top:10px; color:#333;'>
        <strong>TSH (Thyroid Stimulating Hormone) is a key hormone for thyroid analysis.</strong><br>
        This violin plot shows its spread across diagnosis classes: Negative, Hyperthyroid, and Hypothyroid.<br>
        Patients with Hypothyroidism exhibit noticeably higher TSH values, while those with Hyperthyroidism tend to show much lower levels.<br>
        The Negative group has more centralized values, indicating normal TSH distribution in non-thyroid individuals.
    </div>
    """,
    unsafe_allow_html=True
)

        with tab4:
            st.subheader("Thyroid Diagnosis Distribution by Gender")
            st.markdown("📄 **Preview ':**")
            st.dataframe(df.head(20))  # Shows first 20 rows and all columns

            # Preview original columns
            st.markdown("📄 **Preview of 'sex' and 'target':**")
            st.dataframe(df[['sex', 'target']].head(10))
        
            # Step 1: Clean the 'sex' column
            df['sex'] = df['sex'].astype(str).str.upper().str.strip()
            df = df[df['sex'].isin(['M', 'F'])]  # Only keep Male/Female
        
            # Step 2: Clean the 'target' column
            df['target'] = df['target'].astype(str).str.strip().str.title()
            df = df[df['target'].isin(['Negative', 'Hypothyroid', 'Hyperthyroid'])]
        
            # Step 3: Map for display
            df['gender_label'] = df['sex'].map({'F': 'Female', 'M': 'Male'})
        
            # Step 4: Grouped counts
            gender_counts = df.groupby(['gender_label', 'target']).size().reset_index(name='count')
        
            # Preview cleaned data
            st.markdown("✅ **Cleaned 'sex' and 'target' preview:**")
            st.dataframe(gender_counts)
        
            # Step 5: Plot with Plotly
            if gender_counts.empty:
                st.warning("⚠️ No data available after filtering. Please check the 'sex' or 'target' column values.")
            else:
                import plotly.express as px
        
                fig4 = px.bar(
                    gender_counts,
                    x='gender_label',
                    y='count',
                    color='target',
                    barmode='group',
                    title="Thyroid Condition Distribution by Gender",
                    text='count',
                    labels={'gender_label': 'Gender', 'count': 'Number of Patients', 'target': 'Condition'},
                    height=600
                )
                fig4.update_traces(textposition='outside')
                fig4.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig4)
        
                st.caption("This grouped bar chart shows the number of thyroid condition cases by gender (male/female).")

        with tab5:
            st.subheader("TT4 Levels by Diagnosis")
            filtered_tt4 = df[df['TT4'] < 300]
            fig5 = px.box(filtered_tt4, x='target', y='TT4', title='TT4 Distribution by Diagnosis', width=800, height=800)
            st.plotly_chart(fig5)
            st.caption("TT4 (Total Thyroxine) is one of the thyroid hormones. This boxplot shows how its values vary among different diagnostic classes.")

        with tab6:
            st.subheader("Correlation Heatmap of Lab Tests")
            import matplotlib.pyplot as plt
            import seaborn as sns
            corr = df[['TSH', 'T3', 'TT4', 'T4U', 'FTI']].corr()
            fig6, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig6)
            st.caption("This heatmap shows correlation between different thyroid-related lab values. Strong correlation indicates shared diagnostic significance.")

elif page == "About Thyroid":
    st.markdown("""
<style>
.about-section {
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 10px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.about-section h2 {
    color: #2e7d32;
}
.about-section h3 {
    margin-top: 30px;
}
.about-section img {
    max-width: 100%;
    height: auto;
    margin: 10px 0 20px;
}
.references {
    font-size: 15px;
    margin-top: 20px;
    color: #333;
}
.references a {
    color: #3366cc;
    text-decoration: none;
}
.references a:hover {
    text-decoration: underline;
}
</style>

<div class="about-section">
<h2>🦋 Understanding the Thyroid Gland</h2>
<p>The thyroid gland is a small, butterfly-shaped endocrine gland located at the front of the neck, just below the Adam's apple. It plays a crucial role in regulating the body's metabolism by producing hormones that influence various bodily functions, including heart rate, body temperature, and energy levels.The release of thyroid hormones is controlled by a feedback loop involving the hypothalamus and pituitary gland. When thyroid hormone levels are low, the hypothalamus secretes thyrotropin-releasing hormone (TRH), which stimulates the pituitary gland to release thyroid-stimulating hormone (TSH). TSH, in turn, prompts the thyroid to produce more T3 and T4. This system ensures that hormone levels remain balanced, adapting to the body's needs during growth, stress, illness, and changes in environment. </p>
<div style="text-align: center;">
    <img src="https://tse2.mm.bing.net/th?id=OIP.RV3ZwbAm3DkTfGofr2FRVAHaEL&pid=Api" alt="Thyroid Gland Anatomy">
</div>

<h3>Functions of the Thyroid</h3>
<ul>
    <li><strong>Thyroxine (T4)</strong> and <strong>Triiodothyronine (T3)</strong>: Regulate metabolic rate, digestion, brain development, and more.</li>
    <li><strong>Calcitonin</strong>: Regulates calcium levels in the blood.</li>
</ul>

<h3>Common Thyroid Disorders</h3>
<ul>
    <li><strong>Hypothyroidism</strong>: Low hormone production — fatigue, weight gain, etc.</li>
    <li><strong>Hyperthyroidism</strong>: Overactive thyroid — weight loss, rapid heartbeat, etc.</li>
    <li><strong>Goiter</strong>: Enlarged thyroid gland.</li>
    <li><strong>Thyroid Nodules</strong>: Lumps that can be benign or malignant.</li>
    <li><strong>Thyroid Cancer</strong>: Malignant tumors in the thyroid.</li>
</ul>

<h3>Importance of Iodine</h3>
<p>Iodine is vital for thyroid hormone synthesis. Deficiency can lead to goiter or hypothyroidism.</p>

<h3>Treatment for Thyroid Disorders</h3>
<ul>
    <li><strong>Hypothyroidism:</strong> Levothyroxine (synthetic T4).</li>
    <li><strong>Hyperthyroidism:</strong> Antithyroid drugs (e.g., methimazole), radioactive iodine, or surgery.</li>
    <li><strong>Goiter/Nodules:</strong> Iodine supplements, hormone therapy, or surgery based on severity.</li>
    <li><strong>Thyroid Cancer:</strong> Often involves surgery, radioactive iodine, hormone suppression, and possibly chemotherapy.</li>
</ul>
<p>Treatment varies based on severity, patient age, comorbidities, and lab monitoring (e.g., TSH, T3, T4).</p>

<h3>About the Dataset Features</h3>
<p>The dataset includes patient records with clinical and lab features contributing to diagnosis. Key features include:</p>
<ul>
    <li><strong>Age:</strong> Patient's age</li>
    <li><strong>Sex:</strong> Gender (M/F)</li>
    <li><strong>on_thyroxine:</strong> Patient is taking thyroxine</li>
    <li><strong>query_on_thyroxine:</strong> Suspected need for thyroxine</li>
    <li><strong>on_antithyroid_meds:</strong> Taking antithyroid meds</li>
    <li><strong>sick:</strong> Generally sick with non-thyroid illness</li>
    <li><strong>pregnant:</strong> Pregnancy status</li>
    <li><strong>thyroid_surgery:</strong> Previous thyroid surgery</li>
    <li><strong>I131_treatment:</strong> Radioactive iodine therapy</li>
    <li><strong>query_hypothyroid / query_hyperthyroid:</strong> Suspicion of hypo/hyperthyroidism</li>
    <li><strong>lithium:</strong> History of lithium use</li>
    <li><strong>goitre:</strong> Enlarged thyroid</li>
    <li><strong>tumor:</strong> Tumors near/in thyroid</li>
    <li><strong>hypopituitary:</strong> Pituitary issues affecting hormone regulation</li>
    <li><strong>psych:</strong> Psychiatric conditions</li>
    <li><strong>TSH, T3, TT4, T4U, FTI:</strong> Lab blood test values for thyroid hormones</li>
</ul>
<p>These features enable predictive modeling and multi-dimensional thyroid health assessment.</p>

<h3>References</h3>
<div class="references">
    <p>📘 Cleveland Clinic. (n.d.). <em>Thyroid: What It Is, Function & Problems</em>. <br>
    <a href="https://my.clevelandclinic.org/health/body/23188-thyroid" target="_blank">https://my.clevelandclinic.org/health/body/23188-thyroid</a></p>
<div class="references">
    <p>📗 StatPearls. (n.d.). <em>Anatomy, Head and Neck, Thyroid</em>. <br>
    <a href="https://www.ncbi.nlm.nih.gov/books/NBK470452/" target="_blank">https://www.ncbi.nlm.nih.gov/books/NBK470452/</a></p>
<div class="references">
    <p>📙 TeachMeAnatomy. (n.d.). <em>The Thyroid Gland</em>. <br>
    <a href="https://teachmeanatomy.info/neck/viscera/thyroid-gland/" target="_blank">https://teachmeanatomy.info/neck/viscera/thyroid-gland/</a></p>

</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)





with st.expander("About the Model"):
    st.write("""
    - Model: CatBoost Classifier (loaded from final_model.pkl)
    - Trained on 9,000+ patient records from UCI/Kaggle
    - Input: Full feature set used during training
    - Input encoding/normalization is replicated from training logic
    """)
