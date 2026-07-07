import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Delhi House Price Prediction", page_icon="🏠", layout="wide")
st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    height: 70px;        /* Height badhao */
    font-size: 18px;     /* Text bada */
    font-weight: bold;
    border-radius: 12px;
    border: 2px solid #4CAF50;
}
div.stButton > button p {
    font-size: 28px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("house_price_prediction.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏠 Delhi/NCR House Price Prediction")
st.markdown("Predict the estimated price of a house in Delhi/NCR")
st.divider()

cities=["Ghaziabad","Noida","Greater Noida","Faridabad","Gurgaon","New Delhi",
"New Delhi - East","New Delhi - West","New Delhi - South","New Delhi - North",
"New Delhi - Central","New Delhi - Dwarka","New Delhi - Rohini","Gurgaon - South","Gurgaon - North"]
city_coordinates = {
    "Ghaziabad": (28.6692, 77.4538),
    "Greater Noida": (28.4744, 77.5040),
    "Faridabad": (28.4089, 77.3178),
    "Gurgaon": (28.4595, 77.0266),
    "Gurgaon - North": (28.5000, 77.0300),
    "Gurgaon - South": (28.4031, 77.0428),
    "Noida": (28.5355, 77.3910),
    "New Delhi": (28.6139, 77.2090),
    "New Delhi - Central": (28.6304, 77.2177),
    "New Delhi - Dwarka": (28.5921, 77.0460),
    "New Delhi - East": (28.6280, 77.2789),
    "New Delhi - North": (28.7041, 77.1025),
    "New Delhi - Rohini": (28.7495, 77.0565),
    "New Delhi - South": (28.5355, 77.2410),
    "New Delhi - West": (28.6692, 77.0910)
}
left,right=st.columns(2)

with left:
    area=st.number_input("Area (sq.ft)",min_value=500,max_value=10000,value=500)
    bedrooms=st.number_input("Bedrooms",min_value=0,max_value=10,value=0)
    bathrooms=st.number_input("Bathrooms",min_value=0,max_value=10,value=0)
    balcony=st.number_input("Balconies",min_value=0,max_value=10,value=0)
    parking=st.number_input("Parking Spaces",min_value=0,max_value=10,value=0)

with right:
    status=st.selectbox("Status",["Ready to Move","Under Construction"])
    neworold=st.selectbox("Property Type",["Resale","New Property"])
    furnished=st.selectbox("Furnished Status",["Unfurnished","Semi-Furnished","Furnished"])
    building=st.selectbox("Building Type",["Flat","Individual House"])
    city=st.selectbox("City",cities)

if st.button("🏠Predict House Price",use_container_width=True):
    if bedrooms==0 or bathrooms==0 or balcony==0 or parking==0:
        st.error("⚠️ Please fill all property details. Bedrooms, Bathrooms, Balcony and Parking cannot be 0.")
        st.stop()
    input_data=pd.DataFrame([[0]*len(model_columns)],columns=model_columns)
    latitude,longitude=city_coordinates[city]
    input_data["area"]=area
    input_data["latitude"]=latitude
    input_data["longitude"]=longitude
    input_data["Bedrooms"]=bedrooms
    input_data["Bathrooms"]=bathrooms
    input_data["Balcony"]=balcony
    input_data["parking"]=parking
    input_data["Status"]=1 if status=="Ready to Move" else 0
    input_data["neworold"]=0 if neworold=="Resale" else 1
    input_data["Furnished_status"]={"Unfurnished":0,"Semi-Furnished":1,"Furnished":2}[furnished]
    input_data["type_of_building"]=0 if building=="Flat" else 1
    city_column=f"City_{city}"
    if city_column in input_data.columns:
        input_data[city_column]=1
    prediction=model.predict(input_data)[0]
    st.divider()
    st.subheader("📊 Prediction Result")
    st.metric("Estimated Price",f"₹ {prediction:,.0f}")
    st.balloons()
    st.info("This prediction is an estimate based on the trained Machine Learning model.")
   
st.markdown("---")
st.caption("Developed by Hardik Chaturvedi")

st.sidebar.title("🏠 Delhi/NCR House Price Predictor")

st.sidebar.success("Machine Learning Model")

st.sidebar.write("**Algorithm:** Random Forest Regressor")

st.sidebar.write("**Prediction Type:** Regression")

st.sidebar.write("**Location:** Delhi/NCR")

st.sidebar.info(
    "Fill in the property details and click **Predict House Price** to estimate the market price."
)

st.sidebar.markdown("---")

st.sidebar.caption("Developed by Hardik Chaturvedi")