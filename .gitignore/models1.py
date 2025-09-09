from pydantic import BaseModel
from typing import Optional
from datetime import datetime



# class User(BaseModel):
#     username: str
#     email: str
#     password: str  # Plain password (we will hash before saving)

# class Login(BaseModel):
#     email: str
#     password: str

from pydantic import BaseModel, EmailStr

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Detection(BaseModel):
    user_id: str
    item: str
    timestamp: Optional[datetime] = datetime.utcnow()
