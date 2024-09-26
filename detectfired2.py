import streamlit as st
from streamlit_folium import folium_static
import folium  # Add this line
import pandas as pd
import random
import plotly.express as px
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# All 77 Thai provinces with their approximate coordinates

THAI_PROVINCES = {
    "Amnat Charoen": (15.8661, 104.6289),
    "Ang Thong": (14.5896, 100.4549),
    "Bangkok": (13.7563, 100.5018),
    "Bueng Kan": (18.3609, 103.6466),
    "Buri Ram": (14.9951, 103.1116),
    "Chachoengsao": (13.6904, 101.0779),
    "Chai Nat": (15.1851, 100.1251),
    "Chaiyaphum": (15.8068, 102.0317),
    "Chanthaburi": (12.6100, 102.1034),
    "Chiang Mai": (18.7883, 98.9853),
    "Chiang Rai": (19.9105, 99.8406),
    "Chon Buri": (13.3611, 100.9847),
    "Chumphon": (10.4930, 99.1800),
    "Kalasin": (16.4315, 103.5059),
    "Kamphaeng Phet": (16.4827, 99.5226),
    "Kanchanaburi": (14.0023, 99.5328),
    "Khon Kaen": (16.4419, 102.8360),
    "Krabi": (8.0863, 98.9063),
    "Lampang": (18.2854, 99.5122),
    "Lamphun": (18.5743, 99.0087),
    "Loei": (17.4860, 101.7223),
    "Lop Buri": (14.7995, 100.6534),
    "Mae Hong Son": (19.2988, 97.9684),
    "Maha Sarakham": (16.0132, 103.1615),
    "Mukdahan": (16.5425, 104.7240),
    "Nakhon Nayok": (14.2069, 101.2130),
    "Nakhon Pathom": (13.8196, 100.0645),
    "Nakhon Phanom": (17.3948, 104.7692),
    "Nakhon Ratchasima": (14.9799, 102.0978),
    "Nakhon Sawan": (15.7030, 100.1371),
    "Nakhon Si Thammarat": (8.4304, 99.9631),
    "Nan": (18.7756, 100.7730),
    "Narathiwat": (6.4318, 101.8259),
    "Nong Bua Lam Phu": (17.2217, 102.4260),
    "Nong Khai": (17.8782, 102.7418),
    "Nonthaburi": (13.8622, 100.5140),
    "Pathum Thani": (14.0208, 100.5253),
    "Pattani": (6.8691, 101.2550),
    "Phang Nga": (8.4509, 98.5194),
    "Phatthalung": (7.6167, 100.0743),
    "Phayao": (19.2147, 100.2020),
    "Phetchabun": (16.4190, 101.1591),
    "Phetchaburi": (13.1119, 99.9438),
    "Phichit": (16.4398, 100.3489),
    "Phitsanulok": (16.8211, 100.2659),
    "Phra Nakhon Si Ayutthaya": (14.3692, 100.5876),
    "Phrae": (18.1445, 100.1405),
    "Phuket": (7.8804, 98.3923),
    "Prachin Buri": (14.0509, 101.3660),
    "Prachuap Khiri Khan": (11.8126, 99.7957),
    "Ranong": (9.9529, 98.6085),
    "Ratchaburi": (13.5282, 99.8134),
    "Rayong": (12.6815, 101.2816),
    "Roi Et": (16.0566, 103.6517),
    "Sa Kaeo": (13.8244, 102.0645),
    "Sakon Nakhon": (17.1664, 104.1486),
    "Samut Prakan": (13.5990, 100.5998),
    "Samut Sakhon": (13.5475, 100.2745),
    "Samut Songkhram": (13.4094, 100.0021),
    "Saraburi": (14.5289, 100.9109),
    "Satun": (6.6238, 100.0675),
    "Sing Buri": (14.8920, 100.3970),
    "Sisaket": (15.1185, 104.3229),
    "Songkhla": (7.1756, 100.6142),
    "Sukhothai": (17.0069, 99.8265),
    "Suphan Buri": (14.4744, 100.0913),
    "Surat Thani": (9.1351, 99.3268),
    "Surin": (14.8820, 103.4936),
    "Tak": (16.8840, 99.1259),
    "Trang": (7.5645, 99.6239),
    "Trat": (12.2428, 102.5179),
    "Ubon Ratchathani": (15.2448, 104.8472),
    "Udon Thani": (17.4156, 102.7872),
    "Uthai Thani": (15.3838, 100.0255),
    "Uttaradit": (17.6200, 100.0990),
    "Yala": (6.5414, 101.2803),
    "Yasothon": (15.7921, 104.1458)
}

hotspot_lat_long = pd.read_csv('hotspotsugarcane_lat_long.csv')

assumed_day_prediction = [ random.randint(10,30) for _ in range(len(hotspot_lat_long))]

def read_shapefile(uploaded_file):
    # Read the uploaded zipfile
    zip_file = zipfile.ZipFile(uploaded_file)
    
    # Find the .shp file in the zipfile
    shp_file_name = next(name for name in zip_file.namelist() if name.endswith('.shp'))
    
    # Read the shapefile using geopandas
    with zip_file.open(shp_file_name) as shp_file:
        gdf = gpd.read_file(shp_file)
    
    return gdf
def generate_sample_data():
    areas = list(THAI_PROVINCES.keys())
    risks = [random.uniform(0, 100) for _ in range(len(areas))]
    days = [random.randint(1, 16) for _ in range(len(areas))]
    sizes = [random.randint(100, 1000) for _ in range(len(areas))]
    carbon = [size * 8 for size in sizes]  # 8 tons of carbon per rai
    return pd.DataFrame({
        'Area': areas,
        'Risk': risks,
        'Latitude': hotspot_lat_long['LATITUDE'],
        'Longitude': hotspot_lat_long['LONGITUDE'],
        'Day': assumed_day_prediction,
        'Size': sizes,
        'Carbon': carbon
    })

def LSTM_model():
    max_len = 5
    model = Sequential()
    model.add(Embedding(input_dim=10, output_dim=8, input_length=max_len))  # input_dim: ขนาดของ vocab, output_dim: ขนาดของ embedding
    model.add(LSTM(64, return_sequences=True))  # LSTM layer
    model.add(LSTM(32))  # LSTM layer
    model.add(Dense(2, activation='softmax'))  # Output layer สำหรับ 2 คลาส
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model(df):
    X = df[['ndvi_day1', 'ndvi_day2', 'ndvi_day3', ..., 'ndvi_day2x']]
    y = df['burning_days']

    X = pad_sequences(X, maxlen=max_len)
    y = to_categorical(y, num_classes=2)

    model = LSTM_model()
    model.fit(X, y, epochs=5, batch_size=1, validation_split=0.2)

    prediction = model.predict(X)
    print("Predictions:", prediction)

def model_evaluation(df):
    X = df[['NDVI', 'EVI', 'LAI', 'SAVI', 'MSAVI']]
    y = df['burning_days']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #X_train = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    #y_train = np.array([0, 1, 0])  # ตัวอย่าง label สำหรับ 2 คลาส (0 และ 1)
    #X_test = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
    #y_test = np.array([0, 1])
    #padding
    max_len = 5
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    #LSTM
    model = LSTM_model()

    #ฝึกโมเดล
    model.fit(X_train, y_train, epochs=5, batch_size=1, validation_split=0.2)

    

    #ประเมินผลโมเดล
    score = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {score[1]}")

    return score

def main():
    
    st.set_page_config(layout="wide", page_title="Hotspot Prediction Dashboard")
    
    # Custom CSS for improved styling
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    
    .dashboard-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    .risk-item {
        padding: 0.5rem;
        cursor: pointer;
        margin-bottom: 0.5rem;
        border-radius: 0.3rem;
    }
    .risk-high {
        background-color: rgba(255, 0, 0, 0.2);
        border-left: 4px solid red;
    }
    .risk-medium {
        background-color: rgba(255, 165, 0, 0.2);
        border-left: 4px solid orange;
    }
    .risk-low {
        background-color: rgba(0, 128, 0, 0.2);
        border-left: 4px solid green;
    }
    .risk-value {
        font-weight: bold;
    }
    .scrollable-risk-area {
        max-height: 800px;
        overflow-y: auto;
        padding-right: 1rem;
    }
    .stPlotlyChart {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader("Drop shapefile here (as .zip)", type="zip")
        
        if uploaded_file is not None:
            try:
                gdf = read_shapefile(uploaded_file)
                if 'geometry' in gdf.columns:
                    st.session_state.data = gdf
                    st.success("Imported shapefile successfully")
                else:
                    st.error("The shapefile must contain a 'geometry' column.")
            except Exception as e:
                st.error(f"Error reading shapefile: {e}")
        elif 'data' not in st.session_state:
            st.session_state.data = generate_sample_data()
        
        run_button = st.button("Run Analysis")
    
    # Main content
    st.title("Hotspot Prediction")
    
     # Province selection with dropdown
    selected_province = st.selectbox("Choose a province:", ["All Provinces"] + list(THAI_PROVINCES.keys()))

    # Filter data based on selection
    if selected_province and selected_province != "All Provinces":
        filtered_data = st.session_state.data[st.session_state.data['Area'] == selected_province]
        center_lat, center_lon = THAI_PROVINCES[selected_province]
        zoom_start = 16  # Closer zoom for a specific province
    else:
        filtered_data = st.session_state.data
        center_lat, center_lon = 13.7563, 100.5018  # Center of Thailand
        zoom_start = 6  # Default zoom for all of Thailand

    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Map
        st.subheader("Map of Fire Risk Areas")
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
        
        for _, row in filtered_data.iterrows():
            color = 'red' if row['Risk'] > 66 else 'orange' if row['Risk'] > 33 else 'green'
            
            # Add circle with 500m radius (more transparent)
            folium.Circle(
                location=[row['Latitude'], row['Longitude']],
                radius=500,  # 500 meters
                color=color,
                weight=2,  # Slightly thicker border for visibility
                fill=True,
                fill_color=color,
                fill_opacity=0.1,  # Increased transparency
                popup=f"Area: {row['Area']}<br>Risk: {row['Risk']:.2f}%<br>Size: {row['Size']} rai"
            ).add_to(m)
            
            # Add center point
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=3,  # Slightly larger for better visibility
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=1,
                popup=f"Center of {row['Area']}<br>Risk: {row['Risk']:.2f}%"
            ).add_to(m)
        
        folium_static(m, width=800, height=400)
        
        # Amount of net carbon
        st.subheader("Amount of net carbon (8 ton/rai)")
        fig = px.bar(filtered_data, x='Area', y='Carbon', color='Risk',
                     labels={'Carbon': 'Net Carbon (tons)', 'Area': 'Province'},
                     title='Net Carbon by Province',
                     color_continuous_scale='RdYlGn_r')
        fig.update_layout(
            height=400, 
            xaxis={'categoryorder':'total descending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="dashboard-title">Top 20 Fire Risk Areas (Next 16 Days)</h3>', unsafe_allow_html=True)
        st.markdown('<div class="scrollable-risk-area">', unsafe_allow_html=True)
        
        # Sort the filtered data by Risk (descending), Day, and Size
        sorted_data = filtered_data.sort_values(['Risk', 'Day', 'Size'], ascending=[False, True, False])

        # Form State Management
        if 'selected_province' not in st.session_state:
            st.session_state['selected_province'] = None
        
        # Display top 20 areas (or all if less than 20)
        for index, row in sorted_data.head(20).iterrows():
            if row['Risk'] > 66:
                risk_class = "risk-high"
            elif row['Risk'] > 33:
                risk_class = "risk-medium"
            else:
                risk_class = "risk-low"
            
            if st.button(f"({row['Latitude']}, {row['Longitude']}) - Days: {assumed_day_prediction}%"):
                st.session_state.selected_province = row['Area']
        
    if st.session_state.selected_province:
        st.subheader(f"{st.session_state.selected_province} Information")
        with st.form(key='area_form'):
            st.markdown('<div class="dashboard-title">status </div>', unsafe_allow_html=True)
            # Loop through the burn_info dictionary for the selected province
            for index, burn in enumerate(burn_info[st.session_state.selected_province]):
                st.markdown(f"Date: {burn['date']}")
                st.markdown(f"Location: {burn['location']}")
                st.markdown("---")
                submit_button = st.form_submit_button(label=f"Submit{index}")
                if submit_button:
                    st.success(f"Submitted data for {st.session_state['selected_province']}: Size = {500} rai, Risk = {90}%, Days = {1}")


if __name__ == "__main__":
    main()