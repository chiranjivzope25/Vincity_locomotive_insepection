import os
import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Locomotive Axle Lock Early Warning System",
    description="Two-Stage Kinematic & Physical Sensor Fusion API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific domains in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# 1. LOAD ARTIFACTS FROM HUGGING FACE MODEL HUB
# -------------------------------------------------------------


try:
    # Download files from your Hugging Face model repository
   
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Load artifacts using joblib
    model_kinematic = joblib.load(os.path.join(BASE_DIR, "models/axle_lock_xgb.joblib"))
    transformer_kinematic = joblib.load(os.path.join(BASE_DIR, "models/power_transformer.joblib"))

    model_phy = joblib.load(os.path.join(BASE_DIR, "models/phy_axle_lock_xgb.joblib"))
    transformer_phy = joblib.load(os.path.join(BASE_DIR, "models/phy_power_transformer.joblib"))

    print("✅ All ML models and transformers loaded successfully from Hugging Face!")
except Exception as e:
    raise RuntimeError(f"❌ Critical Error loading ML models: {e}")


# -------------------------------------------------------------
# 2. PYDANTIC SCHEMAS
# -------------------------------------------------------------
class KinematicInput(BaseModel):
    v_loco_kmh: float = Field(..., json_schema_extra={"example": 80.0})
    axle1_speed_rads: float = Field(..., json_schema_extra={"example": 55.1})
    axle2_speed_rads: float = Field(..., json_schema_extra={"example": 55.2})
    axle3_speed_rads: float = Field(..., json_schema_extra={"example": 55.0})
    axle4_speed_rads: float = Field(..., json_schema_extra={"example": 54.8})
    axle1_slip_ratio: float = Field(..., json_schema_extra={"example": 0.0})

class PhysicalInput(BaseModel):
    axle1_bearing_temp_c: float = Field(..., json_schema_extra={"example": 105.4})
    axle1_vibration_g: float = Field(..., json_schema_extra={"example": 3.8})
    axle1_motor_current_amp: float = Field(..., json_schema_extra={"example": 520.0})
    
    axle2_bearing_temp_c: float = Field(..., json_schema_extra={"example": 45.0})
    axle2_vibration_g: float = Field(..., json_schema_extra={"example": 0.3})
    axle2_motor_current_amp: float = Field(..., json_schema_extra={"example": 300.0})
    
    axle3_bearing_temp_c: float = Field(..., json_schema_extra={"example": 46.2})
    axle3_vibration_g: float = Field(..., json_schema_extra={"example": 0.35})
    axle3_motor_current_amp: float = Field(..., json_schema_extra={"example": 305.0})
    
    axle4_bearing_temp_c: float = Field(..., json_schema_extra={"example": 44.8})
    axle4_vibration_g: float = Field(..., json_schema_extra={"example": 0.28})
    axle4_motor_current_amp: float = Field(..., json_schema_extra={"example": 298.0})

class DualModelRequest(BaseModel):
    data_axel: KinematicInput
    data_phy: PhysicalInput

# -------------------------------------------------------------
# 3. ENDPOINTS
# -------------------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "Online",
        "system": "Locomotive Axle Lock Dual-Model Inference Service"
    }

@app.post("/predict")
def predict(request: DualModelRequest):
    try:
        # Use dict values directly instead of pandas to save 50MB+ bundle size
        kinematic_features = [list(request.data_axel.model_dump().values())]
        phy_features = [list(request.data_phy.model_dump().values())]
        
        # 1. Transform Features
        x_scaled_kin = transformer_kinematic.transform(kinematic_features)
        x_scaled_phy = transformer_phy.transform(phy_features)
        
        # 2. Get Predictions & Probabilities
        pred_kin = int(model_kinematic.predict(x_scaled_kin)[0])
        prob_kin = float(model_kinematic.predict_proba(x_scaled_kin)[0][1])
        
        pred_phy = int(model_phy.predict(x_scaled_phy)[0])
        prob_phy = float(model_phy.predict_proba(x_scaled_phy)[0][1])
        
        # 3. Safety Rule Override
        speeds = [
            request.data_axel.axle1_speed_rads,
            request.data_axel.axle2_speed_rads,
            request.data_axel.axle3_speed_rads,
            request.data_axel.axle4_speed_rads
        ]
        if request.data_axel.v_loco_kmh > 15.0 and any(s < 5.0 for s in speeds):
            pred_kin = 1
            prob_kin = max(prob_kin, 0.99)

        # 4. Severity Mapping Logic
        if pred_kin == 1 and pred_phy == 1:
            alert_status = "CRITICAL: AXLE LOCK & MECHANICAL SEIZURE CONFIRMED"
            color = "red"
            risk_level = "HIGH"
        elif pred_phy == 1:
            alert_status = "WARNING: HIGH BEARING TEMP / VIBRATION DETECTED"
            color = "orange"
            risk_level = "MEDIUM"
        elif pred_kin == 1:
            alert_status = "CAUTION: WHEEL SLIP OR KINEMATIC LOCK DETECTED"
            color = "yellow"
            risk_level = "LOW-MEDIUM"
        else:
            alert_status = "SYSTEM NORMAL"
            color = "green"
            risk_level = "NORMAL"
            
        return {
            "overall_status": alert_status,
            "display_color": color,
            "risk_level": risk_level,
            "model_outputs": {
                "kinematic_model": {
                    "prediction": pred_kin,
                    "confidence_score": round(prob_kin, 4)
                },
                "physical_model": {
                    "prediction": pred_phy,
                    "confidence_score": round(prob_phy, 4)
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Inference Error: {str(e)}"
        )
