import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "https://serverless.roboflow.com"
API_KEY = "0sr3BL8xb57a3oI6rML1"

WORKSPACE = "dhinesh-m-s-workspace"
WORKFLOW_ID = "detect-count-and-visualize-4"

client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Smart Traffic Management",
    page_icon="🚦",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #F4F7FC;
}

.block-container {
    padding-top: 1rem;
}

.title-box {
    background: linear-gradient(90deg,#0B2C5F,#124E96);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.2);
}

.title-box h1 {
    font-size: 40px;
}

.title-box p {
    font-size: 18px;
    opacity: 0.9;
}

.upload-box {
    background: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.card {
    background: white;
    padding: 15px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.metric-box {
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    color: white;
    margin-bottom: 10px;
}

.green-box {
    background: #16A34A;
}

.orange-box {
    background: #F59E0B;
}

.red-box {
    background: #DC2626;
}

.blue-box {
    background: #2563EB;
}

.decision-box {
    background: linear-gradient(90deg,#0F172A,#1E293B);
    color: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    margin-top: 30px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}

.stSlider {
    padding-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="title-box">
    <h1>🚦 AI SMART TRAFFIC MANAGEMENT SYSTEM</h1>
    <p>
        Vehicle Detection • Traffic Density Analysis • Emergency Vehicle Priority
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# IMAGE UPLOAD
# =========================================================

st.markdown("## 📤 Upload Traffic Road Images")

col1, col2 = st.columns(2)

with col1:

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    road1 = st.file_uploader(
        "Upload Road 1",
        type=["jpg", "jpeg", "png"],
        key="r1"
    )

    road2 = st.file_uploader(
        "Upload Road 2",
        type=["jpg", "jpeg", "png"],
        key="r2"
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    road3 = st.file_uploader(
        "Upload Road 3",
        type=["jpg", "jpeg", "png"],
        key="r3"
    )

    road4 = st.file_uploader(
        "Upload Road 4",
        type=["jpg", "jpeg", "png"],
        key="r4"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PROCESS IMAGE
# =========================================================

def process_image(uploaded_file):

    try:

        uploaded_file.seek(0)

        image = Image.open(uploaded_file).convert("RGB")

        image = image.resize((700, 500))

    except UnidentifiedImageError:

        st.error("❌ Unsupported image format.")
        return None, 0, False, []

    except Exception as e:

        st.error(f"❌ Image Error: {e}")
        return None, 0, False, []

    draw = ImageDraw.Draw(image)

    # =====================================================
    # FONT
    # =====================================================

    try:
        font = ImageFont.truetype("arial.ttf", 16)

    except:
        font = ImageFont.load_default()

    # =====================================================
    # ROBOFLOW INFERENCE
    # =====================================================

    try:

        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": image},
            use_cache=True
        )

    except Exception as e:

        st.error(f"❌ Inference Error: {e}")
        return None, 0, False, []

    # =====================================================
    # SAFE PARSING
    # =====================================================

    try:

        if isinstance(result, list):

            raw = result[0].get("predictions", {})

        else:

            raw = result.get(
                "outputs",
                [{}]
            )[0].get("predictions", {})

        if isinstance(raw, dict):

            predictions = raw.get("predictions", [])

        else:

            predictions = raw

    except:

        predictions = []

    count = 0
    emergency = False
    emergency_types = []

    # =====================================================
    # PROCESS DETECTIONS
    # =====================================================

    for obj in predictions:

        if not isinstance(obj, dict):
            continue

        x = obj.get("x")
        y = obj.get("y")
        w = obj.get("width")
        h = obj.get("height")
        label = obj.get("class")

        if None in (x, y, w, h, label):
            continue

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        label_lower = label.lower()

        # =================================================
        # VEHICLE COLORS
        # =================================================

        if "ambulance" in label_lower:

            color = "#16A34A"
            emergency = True
            emergency_types.append("ambulance")

        elif "fire" in label_lower:

            color = "#F97316"
            emergency = True
            emergency_types.append("fire_truck")

        elif "police" in label_lower:

            color = "#2563EB"
            emergency = True
            emergency_types.append("police")

        elif "bus" in label_lower:

            color = "#9333EA"

        else:

            color = "#EF4444"

        # =================================================
        # DRAW BOX
        # =================================================

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=4
        )

        # =================================================
        # LABEL
        # =================================================

        text = label.replace("_", " ").title()

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        tw = bbox[2] - bbox[0]

        label_y = max(y1 - 30, 5)

        draw.rounded_rectangle(
            [x1, label_y, x1 + tw + 18, label_y + 28],
            radius=6,
            fill=color
        )

        draw.text(
            (x1 + 8, label_y + 5),
            text,
            fill="white",
            font=font
        )

        count += 1

    emergency_types = list(set(emergency_types))

    return image, count, emergency, emergency_types

# =========================================================
# PROCESS ALL ROADS
# =========================================================

road_data = {}

if road1:
    road_data["Road 1"] = process_image(road1)

if road2:
    road_data["Road 2"] = process_image(road2)

if road3:
    road_data["Road 3"] = process_image(road3)

if road4:
    road_data["Road 4"] = process_image(road4)



# =========================================================
# DISPLAY RESULTS
# =========================================================

if road_data:

    st.subheader("📊 Traffic Analysis")

    cols = st.columns(len(road_data))

    for i, (road, data) in enumerate(road_data.items()):

        img, count, emergency, types = data

        with cols[i]:

            # Density Logic
            if count >= 25:
                density = "HIGH"
                box_color = "#DC2626"

            elif count >= 15:
                density = "MEDIUM"
                box_color = "#F59E0B"

            else:
                density = "LOW"
                box_color = "#16A34A"

            # ===============================
            # USE components.html INSTEAD
            # ===============================

            st.components.v1.html(
                f"""
                <div style="
                    background:white;
                    padding:15px;
                    border-radius:15px;
                    box-shadow:0px 2px 8px rgba(0,0,0,0.1);
                    margin-bottom:15px;
                    text-align:center;
                ">

                    <div style="
                        background:{box_color};
                        padding:10px;
                        border-radius:10px;
                        color:white;
                        font-weight:bold;
                        margin-bottom:10px;
                    ">
                        <h2>{road}</h2>
                    </div>

                    <h4>Vehicles Detected</h4>

                    <h1 style="
                        color:#0B2C5F;
                        font-size:42px;
                    ">
                        {count}
                    </h1>

                    <h3 style="color:#374151;">
                        Traffic Density : {density}
                    </h3>

                </div>
                """,
                height=260
            )

            # Show Image
            st.image(
                img,
                use_container_width=True
            )

            # Emergency Alert
            if emergency:

                emergency_text = ", ".join(
                    [t.replace("_", " ").title() for t in types]
                )

                st.error(
                    f"🚑 Emergency Vehicle Detected: {emergency_text}"
                )

            

    # =====================================================
    # EMERGENCY PRIORITY
    # =====================================================

    emergency_roads = {
        r: data[3]
        for r, data in road_data.items()
        if data[2]
    }

    st.markdown("---")

    if emergency_roads:

        st.markdown("## 🚑 Emergency Vehicle Distance Input")

        distance_inputs = {}

        for road, types in emergency_roads.items():

            vehicle_name = ", ".join(types).replace(
                "_",
                " "
            ).title()

            st.markdown(f"""
            <div class="card">
                <h4>{road} - {vehicle_name}</h4>
            </div>
            """, unsafe_allow_html=True)

            distance_inputs[road] = st.slider(
                f"Distance for {road}",
                0,
                500,
                150,
                key=road
            )

        # =================================================
        # PRIORITY RULES
        # =================================================

        priority_map = {
            "ambulance": 1,
            "fire_truck": 2,
            "police": 3
        }

        best_road = None
        best_score = (999, 999)

        for road, types in emergency_roads.items():

            distance = distance_inputs[road]

            for t in types:

                priority = priority_map.get(t, 99)

                if (priority, distance) < best_score:

                    best_score = (
                        priority,
                        distance
                    )

                    best_road = road
                    selected_type = t

        min_distance = best_score[1]

        selected_vehicle = selected_type.replace(
            "_",
            " "
        ).title()

        st.markdown(f"""
        <div class="decision-box">
            🚨 GREEN SIGNAL → {best_road}<br><br>
            {selected_vehicle} Priority<br>
            Distance : {min_distance} meters
        </div>
        """, unsafe_allow_html=True)

    else:

        max_road = max(
            road_data,
            key=lambda r: road_data[r][1]
        )

        max_count = road_data[max_road][1]

        st.markdown(f"""
        <div class="decision-box">
            🚦 GREEN SIGNAL → {max_road}<br><br>
            Maximum Traffic Density<br>
            Vehicles : {max_count}
        </div>
        """, unsafe_allow_html=True)

else:

    st.info("📌 Upload traffic road images to begin analysis.")