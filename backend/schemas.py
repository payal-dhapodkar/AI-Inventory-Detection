from pydantic import BaseModel, EmailStr
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, EmailStr

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserOut(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# --- detection schemas ---
class DetectionOut(BaseModel):
    id: str
    user_id: str
    image_path: Optional[str] = None
    result_json: Dict[str, Any]
    created_at: str

class DetectionList(BaseModel):
    items: List[DetectionOut]
    total: int